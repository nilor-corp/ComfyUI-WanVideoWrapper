import torch
import comfy.utils
from comfy.utils import ProgressBar
from .wanvideo import mm

# Note: The following code is a self-contained block designed to be pasted into `nodes.py`.
# It requires that `nodes.py` correctly imports `get_sigmas`, `sigmas_to_timesteps` from `utils.py`,
# and `get_context_scheduler` from `context.py`.
# You will also need to manually add "TiledWanVideoSampler" to the NODE_CLASS_MAPPINGS.

# --- Local Helper Functions & Classes (to be pasted into nodes.py) ---


def get_embeds(image_embeds, batched_cfg=False, context_window=None):
    """
    Extracts positive and negative embeddings from the main embeds dictionary.
    Handles slicing the embeddings if a context_window is provided.
    """
    if context_window is None:
        positive_embeds = {
            "image_cond": image_embeds.get("image_cond"),
            "clip_fea": image_embeds.get("clip_fea"),
            "text_embeds_f": image_embeds.get("text_embeds_f"),
            "control_latents": image_embeds.get("control_latents"),
            "vace_data": image_embeds.get("vace_data"),
            "unianim_data": image_embeds.get("unianim_data"),
        }
        negative_embeds = {"text_embeds_f": image_embeds.get("text_embeds_f_neg")}
        return positive_embeds, negative_embeds

    c_start, c_end = context_window
    positive_embeds = {
        "image_cond": image_embeds.get("image_cond")[:, :, c_start:c_end, :, :],
        "clip_fea": image_embeds.get("clip_fea")[:, c_start:c_end, :],
        "text_embeds_f": image_embeds.get("text_embeds_f")[:, c_start:c_end, :, :],
        "control_latents": image_embeds.get("control_latents"),
        "vace_data": image_embeds.get("vace_data"),
        "unianim_data": image_embeds.get("unianim_data"),
    }
    negative_embeds = {
        "text_embeds_f": image_embeds.get("text_embeds_f_neg")[:, c_start:c_end, :, :]
    }
    return positive_embeds, negative_embeds


def freenoise_img(x_t, noise, i, T, alpha):
    """Applies freenoise to the latents."""
    if i >= T:
        return x_t
    noise_fft = torch.fft.fft2(noise, norm="ortho")
    x_t_fft = torch.fft.fft2(x_t, norm="ortho")
    band_pass_strength = (1 - (i / T) ** 2) ** 2
    point_filter = (torch.cos(noise_fft) ** 2) * band_pass_strength
    x_t_fft = x_t_fft * (1 - point_filter) + noise_fft * point_filter
    return torch.fft.ifft2(x_t_fft, norm="ortho").real.to(x_t.dtype)


class WindowTracker:
    """A helper class to track context windows."""

    def __init__(self, verbose=False):
        self.windows = {}
        self.verbose = verbose

    def get_window_id(self, frames):
        if frames in self.windows:
            return self.windows[frames]
        else:
            window_id = len(self.windows)
            self.windows[frames] = window_id
            return window_id

    def get_teacache(self, window_id, base_state):
        if window_id not in base_state:
            base_state[window_id] = {}
        return base_state[window_id]


# --- Tiled Sampler Implementation ---


def tiled_predict_with_cfg(
    model,
    tiling_enabled,
    tile_width,
    tile_height,
    tile_padding,
    z,
    cfg_scale,
    positive_embeds,
    negative_embeds,
    timestep,
    idx,
    **kwargs
):
    if not tiling_enabled:
        uncond_pred = model.model.diffusion_model(
            z,
            timestep,
            encoder_hidden_states=negative_embeds["text_embeds_f"],
            **kwargs
        )
        cond_pred = model.model.diffusion_model(
            z,
            timestep,
            encoder_hidden_states=positive_embeds["text_embeds_f"],
            **kwargs
        )
        if isinstance(cfg_scale, (int, float)):
            return uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        return uncond_pred + cfg_scale.view(-1, 1, 1, 1, 1) * (cond_pred - uncond_pred)

    batch_size, channels, frames, height, width = z.shape
    output = torch.zeros_like(z)
    count = torch.zeros_like(z)

    blend_mask = torch.ones((tile_height, tile_width), device=z.device, dtype=z.dtype)
    if tile_padding > 0:
        feather = tile_padding * 2
        for i in range(feather):
            if i < tile_height and i < tile_width:
                value = (i + 1) / feather
                blend_mask[i, :] *= value
                blend_mask[-i - 1, :] *= value
                blend_mask[:, i] *= value
                blend_mask[:, -i - 1] *= value
    blend_mask = blend_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    stride_x = tile_width - tile_padding
    stride_y = tile_height - tile_padding

    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            y_start, y_end = y, min(height, y + tile_height)
            x_start, x_end = x, min(width, x + tile_width)

            z_tile = z[:, :, :, y_start:y_end, x_start:x_end]
            kwargs_tile = kwargs.copy()
            for key in ["image_cond", "control_latents", "vace_data"]:
                if kwargs_tile.get(key) is not None and isinstance(
                    kwargs_tile[key], torch.Tensor
                ):
                    kwargs_tile[key] = kwargs_tile[key][
                        :, :, :, y_start:y_end, x_start:x_end
                    ]

            uncond_pred_tile = model.model.diffusion_model(
                z_tile,
                timestep,
                encoder_hidden_states=negative_embeds["text_embeds_f"],
                **kwargs_tile
            )
            cond_pred_tile = model.model.diffusion_model(
                z_tile,
                timestep,
                encoder_hidden_states=positive_embeds["text_embeds_f"],
                **kwargs_tile
            )

            if isinstance(cfg_scale, (int, float)):
                pred_tile = uncond_pred_tile + cfg_scale * (
                    cond_pred_tile - uncond_pred_tile
                )
            else:
                pred_tile = uncond_pred_tile + cfg_scale.view(-1, 1, 1, 1, 1) * (
                    cond_pred_tile - uncond_pred_tile
                )

            effective_blend_mask = blend_mask[
                :, :, :, : pred_tile.shape[3], : pred_tile.shape[4]
            ]
            output[:, :, :, y_start:y_end, x_start:x_end] += (
                pred_tile * effective_blend_mask
            )
            count[:, :, :, y_start:y_end, x_start:x_end] += effective_blend_mask

    return output / torch.clamp(count, min=1e-6)


class TiledWanVideoSampler:
    @classmethod
    def INPUT_TYPES(s):
        # This requires WanVideoSampler to be defined in the same file
        inputs = WanVideoSampler.INPUT_TYPES()
        inputs["optional"]["tiling_enabled"] = ("BOOLEAN", {"default": False})
        inputs["optional"]["tile_width"] = (
            "INT",
            {"default": 512, "min": 64, "max": 4096, "step": 64},
        )
        inputs["optional"]["tile_height"] = (
            "INT",
            {"default": 512, "min": 64, "max": 4096, "step": 64},
        )
        inputs["optional"]["tile_padding"] = (
            "INT",
            {"default": 128, "min": 0, "max": 1024, "step": 8},
        )
        return inputs

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper"

    def process(
        self,
        model,
        image_embeds,
        shift,
        steps,
        cfg,
        seed,
        scheduler,
        riflex_freq_index,
        text_embeds=None,
        force_offload=True,
        samples=None,
        feta_args=None,
        denoise_strength=1.0,
        context_options=None,
        cache_args=None,
        flowedit_args=None,
        batched_cfg=False,
        slg_args=None,
        rope_function="default",
        loop_args=None,
        experimental_args=None,
        sigmas=None,
        unianimate_poses=None,
        fantasytalking_embeds=None,
        uni3c_embeds=None,
        tiling_enabled=False,
        tile_width=512,
        tile_height=512,
        tile_padding=128,
    ):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model.to(device)

        use_context = context_options is not None
        if use_context:
            context_schedule = context_options.get("context_schedule", "uniform")
            context_frames = (
                (
                    context_options.get(
                        "context_frames", image_embeds.get("image_cond").shape[2] * 4
                    )
                    // 4
                )
                if image_embeds.get("image_cond") is not None
                else 16
            )
            context_stride = context_options.get("context_stride", 4)
            context_overlap = context_options.get("context_overlap", 4)
            context_fade = context_options.get("context_fade", 0.25)
            use_freenoise = context_options.get("freenoise", False)
            verbose = context_options.get("verbose", False)
        else:
            (
                context_schedule,
                context_stride,
                context_overlap,
                context_fade,
                use_freenoise,
                verbose,
            ) = ("uniform", 1, 0, 0.25, False, False)
            context_frames = (
                (image_embeds.get("image_cond").shape[2] * 4)
                if image_embeds.get("image_cond") is not None
                else 16
            )

        latent_shift_skip = loop_args.get("shift_skip", 0) if loop_args else 0
        init_noise = torch.randn_like(samples["samples"]) if use_freenoise else None

        if text_embeds:
            image_embeds["text_embeds_f"] = text_embeds
        if "text_embeds_f" not in image_embeds or image_embeds["text_embeds_f"] is None:
            raise ValueError("text_embeds is required")

        latent_video_length = samples["samples"].shape[2]
        if use_context and "context_windows" not in image_embeds:
            # This is the corrected logic that uses the existing get_context_scheduler
            context_scheduler = get_context_scheduler(context_schedule)
            image_embeds["context_windows"] = list(
                context_scheduler(
                    num_frames=latent_video_length,
                    context_size=context_frames // 4,
                    context_stride=context_stride,
                    context_overlap=context_overlap // 4,
                    closed_loop=(context_schedule == "uniform_looped"),
                )
            )

        # These functions need to be imported from utils.py
        loaded_sigmas = (
            sigmas
            if sigmas is not None
            else get_sigmas(scheduler, steps, shift, riflex_freq_index)
        )
        timesteps = sigmas_to_timesteps(loaded_sigmas, steps)

        z = samples["samples"].to(device)

        if z.shape[2] != latent_video_length:
            z = torch.nn.functional.interpolate(
                z.permute(0, 2, 1, 3, 4),
                (latent_video_length, z.shape[3], z.shape[4]),
                mode="trilinear",
            ).permute(0, 2, 1, 3, 4)
        if (
            image_embeds.get("image_cond") is not None
            and z.shape[3:] != image_embeds["image_cond"].shape[3:]
        ):
            z = (
                torch.nn.functional.interpolate(
                    z.flatten(0, 2).unsqueeze(0),
                    image_embeds["image_cond"].shape[3:],
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .reshape(
                    z.shape[0],
                    z.shape[1],
                    z.shape[2],
                    image_embeds["image_cond"].shape[3],
                    image_embeds["image_cond"].shape[4],
                )
            )

        z = (
            z + torch.randn_like(z) * loaded_sigmas[0]
            if denoise_strength < 1.0
            else torch.randn_like(z) * loaded_sigmas[0]
        )

        positive_embeds, negative_embeds = get_embeds(image_embeds, batched_cfg=False)
        audio_proj = (
            fantasytalking_embeds.get("audio_proj") if fantasytalking_embeds else None
        )
        if audio_proj:
            positive_embeds["audio_proj"] = audio_proj
            negative_embeds["audio_proj"] = audio_proj.clone()
        control_camera_latents = (
            uni3c_embeds.get("control_camera_latents") if uni3c_embeds else None
        )

        cache_state = {}
        pbar = ProgressBar(steps)
        if rope_function == "comfy":
            model.model.set_rope("comfy")

        for i, t in enumerate(timesteps):
            if use_freenoise:
                z = freenoise_img(z, init_noise, i, steps, 1.0)

            predict_kwargs = {
                "image_cond": positive_embeds.get("image_cond"),
                "clip_fea": positive_embeds.get("clip_fea"),
                "control_latents": positive_embeds.get("control_latents"),
                "vace_data": positive_embeds.get("vace_data"),
                "unianim_data": unianimate_poses,
                "audio_proj": audio_proj,
                "control_camera_latents": control_camera_latents,
            }

            noise_pred = tiled_predict_with_cfg(
                model,
                tiling_enabled,
                tile_width,
                tile_height,
                tile_padding,
                z,
                cfg,
                positive_embeds,
                negative_embeds,
                t,
                i,
                **predict_kwargs
            )

            if (
                latent_shift_skip > 0
                and i > 0
                and i % latent_shift_skip == 0
                and i < steps - 1
            ):
                z = torch.cat((z[:, :, 1:], z[:, :, -1:]), dim=2)
                noise_pred = torch.cat(
                    (noise_pred[:, :, 1:], noise_pred[:, :, -1:]), dim=2
                )

            z = z - noise_pred * (loaded_sigmas[i] - loaded_sigmas[i + 1])
            pbar.update(1)

        if rope_function == "comfy":
            model.model.set_rope("default")
        if force_offload:
            model.to(offload_device)
            mm.soft_empty_cache()
        return ({"samples": z.to(mm.unet_offload_device())},)
