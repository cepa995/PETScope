import os
import ants
from src.logger import logger

def ants_registration(moving_img_path, fixed_img_path, registration_dir, 
                      filename_moving_to_fixed, filename_fixed_to_moving, 
                      type_of_transform='Rigid'):
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
    # Ensure the provided image paths exist
    for img_path in [moving_img_path, fixed_img_path]:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"{img_path} does not exist")
    
    os.makedirs(registration_dir, exist_ok=True)

    # Read images
    logger.info(f"Reading Moving Image: {moving_img_path}")
    logger.info(f"Reading Fixed Image: {fixed_img_path}")
    moving_img = ants.image_read(moving_img_path)
    fixed_img = ants.image_read(fixed_img_path)

    # Register PET to T1
    logger.info(f"Start Registration: {os.path.basename(moving_img_path)} to {os.path.basename(fixed_img_path)}")
    registration = ants.registration(
        fixed_img, moving_img,
        reg_iterations=[110, 110, 30],
        type_of_transform=type_of_transform,
        outprefix=os.path.join(registration_dir, 'reg_pet_to_t1_')
    )

    # Warp images
    logger.info("Warp MOVING image to FIXED image space")
    warped_moving = ants.apply_transforms(fixed_img, moving_img, transformlist=registration['fwdtransforms'])

    logger.info("Warp FIXED image to MOVING image space (inverse)")
    warped_fixed = ants.apply_transforms(moving_img, fixed_img, transformlist=registration['invtransforms'], whichtoinvert=[True])

    # Write the warped images
    ants.image_write(warped_moving, os.path.join(registration_dir, filename_moving_to_fixed))
    ants.image_write(warped_fixed, os.path.join(registration_dir, filename_fixed_to_moving))

    # Return the transformation file path
    transform = [os.path.join(registration_dir, f) for f in os.listdir(registration_dir) if f.endswith('.mat')][0]
    return transform

def ants_warp_image(fixed_image_path, moving_image_path, transform_path, output_path, is_inverse=False, interpolator='linear'):
    """
    Warps moving image to a fixed image space.

    :param fixed_image_path: absolute path to a fixed image
    :param moving_image_path: absolute path to a moving image
    :param transform_path: absolute path to transform file from ANTs registration
    :param output_path: absolute path to the resulting warp image
    :param is_inverse: if the transform matrix should be inverse or not (default: False)
    :param interpolator: method of interpolation (default is 'linear')
    :returns: warped image (Nifti1Image object)
    """
    # Read the images
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)

    # Apply the transformation
    warped_image = ants.apply_transforms(
        fixed_image,
        moving_image, 
        transformlist=[transform_path],
        whichtoinvert=[is_inverse], 
        interpolator=interpolator
    )

    # Write and return the warped image
    ants.image_write(warped_image, output_path)
    return warped_image
