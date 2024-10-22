import os
import ants
from rich import print
from petscope.logger import logger
from petscope.constants import SUPPORTED_ANTS_TRANSFORM_TYPES

def ants_registration(
    moving_img_path: str, 
    fixed_img_path: str, 
    registration_dir: str, 
    filename_moving_to_fixed: str, 
    filename_fixed_to_moving: str, 
    type_of_transform: str = 'Rigid'
) -> str:
    """
    Utilize ANTs registration to move PET image into T1 space.

    :param moving_img_path: absolute path to PET 3D volume
    :param fixed_img_path: absolute path to T1 3D volume
    :param registration_dir: absolute path to resulting directory
    :param filename_moving_to_fixed: filename of PET image in T1 space
    :param filename_fixed_to_moving: filename of T1 image in PET space
    :param type_of_transform: type of transformation (default is 'Rigid')
    :returns: transformation file path and inverse transformation file path
    """
    # Ensure valid registration type has been passed
    if type_of_transform not in SUPPORTED_ANTS_TRANSFORM_TYPES:
        from petscope.exceptions import ANTsUnsupportedTransformType
        raise ANTsUnsupportedTransformType(
            f"{type_of_transform} is not a valid transformation type. "
            f"Please choose one of the following {SUPPORTED_ANTS_TRANSFORM_TYPES}"
        )

    # Ensure the provided image paths exist
    for img_path in [moving_img_path, fixed_img_path]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} does not exist")
    
    os.makedirs(registration_dir, exist_ok=True)

    try:
        # Read images
        print(f"\t[bold blue][READING]: [green] Moving Image {moving_img_path}")
        moving_img = ants.image_read(moving_img_path)
        print(f"\t[bold blue][READING]: [green] Fixed Image {fixed_img_path}")
        fixed_img = ants.image_read(fixed_img_path)
    except Exception:
        from petscope.exceptions import ANTsImageReadException
        raise ANTsImageReadException("Path to the images is valid, but the type might not be supported by ANTs")
    
    try:
        # Register PET to T1
        print(f"\t[bold blue][REGISTRATION]: [bold green]{os.path.basename(moving_img_path)} to {os.path.basename(fixed_img_path)}")
        registration = ants.registration(
            fixed_img, moving_img,
            reg_iterations=[110, 110, 30],
            type_of_transform=type_of_transform,
            outprefix=os.path.join(registration_dir, 'reg_')
        )
    except Exception:
        from petscope.exceptions import ANTsRegistrationException
        raise ANTsRegistrationException(f"Exception occurred while trying to register {moving_img_path} to {fixed_img_path}")
    
    try:
        # Warp images
        print("\t[bold blue][APPLY TRANSFORM]: [green]Moving image to Fixed image space")
        warped_moving = ants.apply_transforms(fixed_img, moving_img, transformlist=registration['fwdtransforms'])

        print("\t[bold blue][APPLY INV TRANSFORM]: [green]Fixed image to Moving image space")
        warped_fixed = ants.apply_transforms(moving_img, fixed_img, transformlist=registration['invtransforms'], whichtoinvert=[True])
    except Exception:
        from petscope.exceptions import ANTsApplyTransformsException
        raise ANTsApplyTransformsException("Exception occurred while trying to apply ANTs transformation")
    
    # Write the warped images
    ants.image_write(warped_moving, os.path.join(registration_dir, filename_moving_to_fixed))
    ants.image_write(warped_fixed, os.path.join(registration_dir, filename_fixed_to_moving))

    # Return the transformation file path
    transform = [os.path.join(registration_dir, f) for f in os.listdir(registration_dir) if f.endswith('.mat')][0]
    print("\t:white_heavy_check_mark: [bold green]SUCCESS! ")
    return transform

def ants_warp_image(
    fixed_img_path: str, 
    moving_img_path: str, 
    transform_path: str, 
    output_path: str, 
    is_inverse: bool = False, 
    interpolator: str = 'linear'
) -> ants.core.ants_image.ANTsImage:
    """
    Warps moving image to a fixed image space.

    :param fixed_img_path: absolute path to a fixed image
    :param moving_img_path: absolute path to a moving image
    :param transform_path: absolute path to transform file from ANTs registration
    :param output_path: absolute path to the resulting warp image
    :param is_inverse: if the transform matrix should be inverse or not (default: False)
    :param interpolator: method of interpolation (default is 'linear')
    :returns: warped image (ANTsImage object)
    """
    try:
        # Read the images
        fixed_image = ants.image_read(fixed_img_path)
        moving_image = ants.image_read(moving_img_path)
    except Exception:
        from petscope.exceptions import ANTsImageReadException
        raise ANTsImageReadException("Path to the images is valid, but the type might not be supported by ANTs")
    
    try:
        # Apply the transformation
        warped_image = ants.apply_transforms(
            fixed_image,
            moving_image, 
            transformlist=[transform_path],
            whichtoinvert=[is_inverse], 
            interpolator=interpolator
        )
    except Exception:
        from petscope.exceptions import ANTsApplyTransformsException
        raise ANTsApplyTransformsException(
            f"Could not apply {transform_path} to {moving_img_path} where fixed image is {fixed_img_path}"
        )

    # Write and return the warped image
    ants.image_write(warped_image, output_path)
    return warped_image
