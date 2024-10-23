import os

from src.registration import ants_warp_image, ants_registration
from src.utils import convert_4d_to_3d, compute_mean_volume, compute_4d_image, change_dtype
from src.logger import logger
from src.tests.data import data
from src.tests.utils import init_argparse

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()

    # Step 0. Split 4D to 3D volumes
    pet_3d_vol_dir = os.path.join(args.test_results_dir, "3d_volumes")
    logger.info("Convert 4D PET image into list of 3D Volumes")
    convert_4d_to_3d(
        img_4d_path=data['PET4D'],
        img_3d_dir=pet_3d_vol_dir,
        orientation='RSA'
    )

    # Step 1. Compute PET 3D volume
    logger.info("Computing Mean 3D Volume")
    pet_3d_volume = os.path.join(args.test_results_dir, 'pet_3d.nii')
    pet_4d_volume = os.path.join(args.test_results_dir, 'pet_4d_rsa.nii')
    mean_3d_image = compute_mean_volume(pet_3d_vol_dir, pet_3d_volume)
    pet_4d_rsa = compute_4d_image(pet_3d_vol_dir, pet_4d_volume)

    # Step 2. Register PET to T1
    logger.info("Rigid Registration: PET -> T1 Space")
    registration_dir = os.path.join(args.test_results_dir, "PET_TO_T1")
    transform = ants_registration(
        moving_img_path=pet_3d_volume,
        fixed_img_path=data['T13D'],
        registration_dir=registration_dir,
        filename_fixed_to_moving='t1_pet_space.nii',
        filename_moving_to_fixed='pet_t1_space.nii',
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
