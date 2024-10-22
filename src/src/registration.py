import os
import ants

from scripts.logger import logger

def pet_to_t1(pet_3d_path, t1_3d_path, registration_dir, 
              output_filename_pet_t1_space, output_filename_t1_pet_space, 
              type_of_transform='Rigid'):
    """
    Utilize ANTs registration to move PET image into T1 space

    :param pet_3d_path - absolute path to PET 3D volume
    :param t1_3d_path - absolute path to T1 3D Volume
    :param registration_dir - absolute path to resulting directory
    :param output_filename_pet_t1_space - filename of PET image in T1 space
    :param output_filename_t1_pet_space - filename of T1 image in PET space
    :returns transformation file path and inverse transformation file path
    """
    for img_path in [pet_3d_path, t1_3d_path]:
        if not os.path.exists(img_path):
            raise FileNotFoundError("{} does not exist".format(img_path))
    os.makedirs(registration_dir, exist_ok=True)
    # Read images
    pet_3d_img = ants.image_read(pet_3d_path)
    t1_3d_img  = ants.image_read(t1_3d_path)
    # Register PET to T1
    registration  = ants.registration(
        t1_3d_img, pet_3d_img,
        reg_iterations=[110,110,30],
        type_of_transform=type_of_transform,
        outprefix=os.path.join(registration_dir, 'reg_pet_to_t1_')
    )
    warped_pet_3d = ants.apply_transforms(t1_3d_img, pet_3d_img, transformlist=registration['fwdtransforms'])
    warped_t1_3d  = ants.apply_transforms(pet_3d_img, t1_3d_img, transformlist=registration['invtransforms'], whichtoinvert=[True])
    ants.image_write(warped_pet_3d, os.path.join(registration_dir, output_filename_pet_t1_space))
    ants.image_write(warped_t1_3d, os.path.join(registration_dir, output_filename_t1_pet_space))
    # Return image transformation files
    transform     = [os.path.join(registration_dir, f) for f in os.listdir(registration_dir) if f.endswith('.mat')][0]
    return transform

def ants_warp_image(fixed_image_path, moving_image_path, transform_path, output_path, is_inverse=False, interpolator='linear'):
    """
    Warps moving image to a fixed image space

    :param fixed_image_path - absolute path to a fixed image
    :param moving_image_path - absolute path to a moving image
    :param is_inverse - if the transform matrix should be inverse or not
      (default: False)
    :param transform_path - absolute path to transform file from
      ANTs registration
    :param output_path - absolute path to the resulting warp image
    :returns warped image (Nifti1Image object)
    """
    fixed_image     = ants.image_read(fixed_image_path)
    moving_image    = ants.image_read(moving_image_path)
    warped_image = ants.apply_transforms(
        fixed_image,
        moving_image, 
        transformlist=[transform_path],
        whichtoinvert=[is_inverse], 
        interpolator=interpolator
    )
    ants.image_write(warped_image, output_path)
    return warped_image
