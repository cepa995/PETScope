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
    Performs image registration using ANTs to align a moving image (e.g., PET) to a fixed image (e.g., T1).

    This function reads the moving and fixed images, performs the specified registration transformation,
    and applies forward and inverse transformations to warp the images. The resulting images are saved
    in the specified directory.

    Args:
        moving_img_path (str): Absolute path to the moving image (e.g., PET 3D volume).
        fixed_img_path (str): Absolute path to the fixed image (e.g., T1 3D volume).
        registration_dir (str): Directory to save registration results.
        filename_moving_to_fixed (str): Filename for the moving image warped to the fixed image space.
        filename_fixed_to_moving (str): Filename for the fixed image warped to the moving image space.
        type_of_transform (str): Type of transformation to perform (e.g., "Rigid", "Affine"). Default is "Rigid".

    Returns:
        str: Path to the forward transformation file generated during registration.

    Raises:
        FileNotFoundError: If the specified image paths do not exist.
        ANTsUnsupportedTransformType: If the specified transform type is not supported by ANTs.
        ANTsImageReadException: If the images cannot be read by ANTs.
        ANTsRegistrationException: If an error occurs during the registration process.
        ANTsApplyTransformsException: If an error occurs while applying the transformations.

    Example:
        transform_path = ants_registration(
            moving_img_path="/path/to/pet_image.nii",
            fixed_img_path="/path/to/t1_image.nii",
            registration_dir="/path/to/output",
            filename_moving_to_fixed="pet_in_t1_space.nii",
            filename_fixed_to_moving="t1_in_pet_space.nii",
            type_of_transform="Rigid"
        )
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
    Applies a transformation to warp a moving image into the fixed image's space using ANTs.

    This function reads the fixed and moving images, applies the provided transformation (or its inverse),
    and saves the resulting warped image to the specified output path.

    Args:
        fixed_img_path (str): Absolute path to the fixed image.
        moving_img_path (str): Absolute path to the moving image.
        transform_path (str): Absolute path to the transformation file generated by ANTs registration.
        output_path (str): Absolute path to save the resulting warped image.
        is_inverse (bool, optional): Whether to apply the inverse of the transformation. Default is False.
        interpolator (str, optional): Interpolation method (e.g., "linear", "nearest"). Default is "linear".

    Returns:
        ants.core.ants_image.ANTsImage: The warped image as an ANTsImage object.

    Raises:
        ANTsImageReadException: If the images cannot be read by ANTs.
        ANTsApplyTransformsException: If an error occurs while applying the transformation.

    Example:
        warped_image = ants_warp_image(
            fixed_img_path="/path/to/fixed_image.nii",
            moving_img_path="/path/to/moving_image.nii",
            transform_path="/path/to/transform.mat",
            output_path="/path/to/warped_image.nii",
            is_inverse=False,
            interpolator="linear"
        )
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
