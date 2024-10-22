import os

from scripts.registration import ants_warp_image, pet_to_t1
from scripts.utils import convert_4d_to_3d, compute_pet_mean_and_4d_volumes
from scripts.logger import logger
from scripts.tests.data import data

if __name__ == "__main__":
    # Step 0. Split 4D to 3D volumes
    pet_3d_vol_dir = os.path.join(data['RESULTS_DIR'], "3d_volumes")
    logger.info("Convert 4D PET image into list of 3D Volumes")
    convert_4d_to_3d(
        img_4d_path=data['PET4D'],
        img_3d_dir=pet_3d_vol_dir,
        orientation='RSA'
    )
    # Step 1. Compute PET 3D volume
    logger.info("Computing Mean 3D Volume")
    pet_3d_volume = os.path.join(data['RESULTS_DIR'], 'pet_3d.nii')
    pet_4d_volume = os.path.join(data['RESULTS_DIR'], 'pet_4d_rsa.nii')
    mean_3d_image, pet_4d_rsa = compute_pet_mean_and_4d_volumes(pet_3d_vol_dir, pet_3d_volume, pet_4d_volume)
    # Step 2. Register PET to T1
    logger.info("Rigid Registration: PET -> T1 Space")
    registration_dir = os.path.join(data['RESULTS_DIR'], "PET_TO_T1")
    transform = pet_to_t1(
        pet_3d_path=pet_3d_volume,
        t1_3d_path=data['T13D'],
        registration_dir=registration_dir,
        output_filename_t1_pet_space='t1_pet_space.nii',
        output_filename_pet_t1_space='pet_t1_space.nii',
        type_of_transform='Rigid'
    )
    # Step 3. Move brain mask from T1 to PET
    brain_mask_pet_space = os.path.join(registration_dir, 'brain_mask_pet_space.nii')
    ants_warp_image(
        fixed_image_path=data['BRAIN_MASK'],
        moving_image_path=pet_3d_volume,
        transform_path=transform,
        is_inverse=True,
        interpolator='genericLabel',
        output_path=brain_mask_pet_space
    )