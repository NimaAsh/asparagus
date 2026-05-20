from asparagus.modules.transforms import Torch_ClampTarget
from gardening_tools.functional.transforms.spatial import get_max_rotated_size
from gardening_tools.modules.transforms.bias_field import Torch_BiasField
from gardening_tools.modules.transforms.blur import Torch_Blur
from gardening_tools.modules.transforms.copy_image_to_label import Torch_CopyImageToLabel
from gardening_tools.modules.transforms.cropping_and_padding import Torch_CropPad
from gardening_tools.modules.transforms.gamma import Torch_Gamma
from gardening_tools.modules.transforms.masking import Torch_Mask
from gardening_tools.modules.transforms.motion_ghosting import Torch_MotionGhosting
from gardening_tools.modules.transforms.noise import Torch_AdditiveNoise, Torch_MultiplicativeNoise
from gardening_tools.modules.transforms.normalize import Torch_Normalize
from gardening_tools.modules.transforms.ringing import Torch_GibbsRinging
from gardening_tools.modules.transforms.sampling import Torch_SimulateLowres
from gardening_tools.modules.transforms.spatial import Torch_Spatial
from torchvision import transforms


def CPU_val_transforms(patch_size):
    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_CropPad(patch_size=patch_size, p_oversample_foreground=0.0),
            Torch_CopyImageToLabel(copy=True),
            Torch_ClampTarget(clamp=True, min_value=-2.0, max_value=4.0),
        ]
    )


def CPU_train_transforms(patch_size):
    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = get_max_rotated_size(patch_size)
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_CropPad(patch_size=pre_aug_patch_size, p_oversample_foreground=0.0),
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                p_scale_all_channel=p_scale_all_channel,
                clip_to_input_range=False,
                skip_label=False,
            ),
            Torch_CopyImageToLabel(copy=True),
            Torch_ClampTarget(clamp=True, min_value=-2.0, max_value=4.0),
        ]
    )


def GPU_train_transforms(masking=False, ndim=3, mask_ratio=0.6, patch_size=None):
    # `patch_size` kwarg is accepted for signature compatibility with
    # `GPU_train_transforms_with_spatial`. It is unused here because the CPU
    # transforms produce already-cropped patches.
    del patch_size
    axes = (0, ndim)
    tforms = transforms.Compose(
        [
            Torch_Blur(p_per_channel=0.1),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.2),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.1, p_per_axis=0.3),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if masking:
        tforms.transforms.append(Torch_Mask(ratio=mask_ratio))

    return tforms


def GPU_val_transforms(masking=False, mask_ratio=0.6, patch_size=None):
    # `patch_size` kwarg is accepted for signature compatibility with the
    # `_with_spatial` train variant. It is unused on the val path.
    del patch_size
    if masking:
        return transforms.Compose(
            [
                Torch_Mask(ratio=mask_ratio),
            ]
        )
    return None


def CPU_train_transforms_lite(patch_size):
    """CPU-side transforms when Torch_Spatial is run on GPU.

    Keeps the pre-aug crop so the worker only does load + normalize + crop
    (cheap) and lets the heavy grid_sample run on the GPU as part of
    `on_after_batch_transfer`.
    """

    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2

    if p_rot_all_channel > 0 or p_scale_all_channel > 0:
        pre_aug_patch_size = get_max_rotated_size(patch_size)
    else:
        pre_aug_patch_size = patch_size

    return transforms.Compose(
        [
            Torch_Normalize(normalize=True),
            Torch_CropPad(patch_size=pre_aug_patch_size, p_oversample_foreground=0.0),
        ]
    )


def GPU_train_transforms_with_spatial(masking=False, ndim=3, mask_ratio=0.6, patch_size=None):
    """GPU train transforms that include rotation/scale before the rest.

    Caveat: a single grid is generated per batch, so every sample in the batch
    receives the same rotation/scale. This trades per-sample augmentation
    diversity for ~all of the wall-clock improvement (Torch_Spatial is the
    dominant CPU-worker cost for MAE-style pretraining).

    The flow is: Spatial(skip_label=True) -> CopyImageToLabel -> ClampTarget
    -> standard photometric augs -> Mask. Skip_label=True is correct because
    CPU_train_transforms_lite does not create a label; the copy happens after
    the spatial transform, matching the original "label = clean spatial
    image" semantics.
    """

    if patch_size is None:
        raise ValueError("GPU_train_transforms_with_spatial requires patch_size")

    p_rot_all_channel = 0.2
    p_scale_all_channel = 0.2
    axes = (0, ndim)

    tforms = transforms.Compose(
        [
            Torch_Spatial(
                patch_size=patch_size,
                p_deform_all_channel=0.0,
                p_rot_all_channel=p_rot_all_channel,
                p_rot_per_axis=0.3,
                p_scale_all_channel=p_scale_all_channel,
                clip_to_input_range=False,
                skip_label=True,
            ),
            Torch_CopyImageToLabel(copy=True),
            Torch_ClampTarget(clamp=True, min_value=-2.0, max_value=4.0),
            Torch_Blur(p_per_channel=0.1),
            Torch_BiasField(p_per_channel=0.2),
            Torch_Gamma(p_all_channel=0.2),
            Torch_MotionGhosting(p_per_channel=0.1, axes=axes),
            Torch_GibbsRinging(p_per_channel=0.1, axes=axes),
            Torch_SimulateLowres(p_per_channel=0.1, p_per_axis=0.3),
            Torch_MultiplicativeNoise(p_per_channel=0.1),
            Torch_AdditiveNoise(p_per_channel=0.1),
        ]
    )

    if masking:
        tforms.transforms.append(Torch_Mask(ratio=mask_ratio))

    return tforms
