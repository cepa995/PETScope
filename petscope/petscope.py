import os
from rich import print
from petscope.dynamicpet_wrapper.srtm import call_srtm
from petscope.registration import ants_registration, ants_warp_image
from petscope.utils import convert_4d_to_3d, compute_mean_volume, compute_4d_image, c3d_space_check
from petscope.petpvc_wrapper.utils import petpvc_create_4d_mask
from petscope.petpvc_wrapper.petpvc import run_petpvc_iterative_yang

class PETScope:
    def __init__(self) -> None:
        pass

    def pet_to_t1(
        self,
        pet_4d_path: str,
        t1_3d_path: str,
        type_of_transform: str,
        output_dir: str
    ) -> int:
        # Convert 4D PET image to sequence of 3D volumes
        print(":gear: STEP 1. [bold green]Converting 4D PET to Sequence of 3D Volumes")
        pet_3d_volumes_dir = os.path.join(output_dir, "pet_3d_volumes")
        os.makedirs(pet_3d_volumes_dir, exist_ok=True)
        convert_4d_to_3d(
            img_4d_path=pet_4d_path,
            img_3d_dir=pet_3d_volumes_dir,
            orientation='RSA'
        )

        # Compute PET 3D Mean Volume
        print(":gear: STEP 2. [bold green]Computing MEAN 3D Volume")
        pet_3d_mean_volume_path = os.path.join(output_dir, 'pet_3d_mean.nii')
        _ = compute_mean_volume(
            volume_dir=pet_3d_volumes_dir,
            mean_3d_out=pet_3d_mean_volume_path
        )

        # Rigid Registration - PET to T1 Space
        print(f":gear: STEP 3. [bold green]Running ANTs {type_of_transform} Registration")
        registration_dir = os.path.join(output_dir, 'pet_to_t1_registration')
        os.makedirs(registration_dir, exist_ok=True)
        _ = ants_registration(
            moving_img_path=pet_3d_mean_volume_path,
            fixed_img_path=t1_3d_path,
            registration_dir=registration_dir,
            filename_fixed_to_moving='t1_pet_space.nii',
            filename_moving_to_fixed='pet_t1_space.nii',
            type_of_transform=type_of_transform
        )

        return 0

    def run_srtm(
        self,
        pet_4d_path: str,
        t1_3d_path: str,
        template_path: str,
        template: str,
        reference_region: str,
        output_dir: str,
        model: str = 'SRTMZhou2003'
    ):
        # TODO: Validation:
        # 1. Check if T1 Image is in the same space as Template image: DONE
        # Add more validations for the Input
        print(":gear: STEP 0. [bold green]Validating Input Arguments")
        if not c3d_space_check(template_path, t1_3d_path):
            from petscope.exceptions import NotSamePhysicalSpaceException
            raise NotSamePhysicalSpaceException(
                f"Template image {template_path} is not in the same space as T1 image {t1_3d_path}"
            )
        print("\t:white_heavy_check_mark: [bold green]INPUTS ARE VALID!")

        # Convert 4D PET image to sequence of 3D volumes
        print(":gear: STEP 1. [bold green]Converting 4D PET to Sequence of 3D Volumes")
        pet_3d_volumes_dir = os.path.join(output_dir, "pet_3d_volumes")
        os.makedirs(pet_3d_volumes_dir, exist_ok=True)
        convert_4d_to_3d(
            img_4d_path=pet_4d_path,
            img_3d_dir=pet_3d_volumes_dir,
            orientation='RSA'
        )

        # Compute PET 3D Mean Volume
        print(":gear: STEP 2. [bold green]Computing MEAN 3D Volume")
        pet_3d_mean_volume_path = os.path.join(output_dir, 'pet_3d_mean.nii')
        _ = compute_mean_volume(
            volume_dir=pet_3d_volumes_dir,
            mean_3d_out=pet_3d_mean_volume_path
        )

        # Rigid Registration - PET to T1 Space
        print(":gear: STEP 3. [bold green]Running ANTs Rigid Registration")
        registration_dir = os.path.join(output_dir, 'pet_to_t1_registration')
        os.makedirs(registration_dir, exist_ok=True)
        transformation_path = ants_registration(
            moving_img_path=pet_3d_mean_volume_path,
            fixed_img_path=t1_3d_path,
            registration_dir=registration_dir,
            filename_fixed_to_moving='t1_pet_space.nii',
            filename_moving_to_fixed='pet_t1_space.nii',
            type_of_transform='Rigid'
        )

        # Warp reference mask to PET space
        print(":gear: STEP 4. [bold green]Warp Reference Mask (MNI) to PET Space")
        template_pet_space_path = os.path.join(registration_dir, 'template_pet_space.nii')
        ants_warp_image(
            fixed_img_path=pet_3d_mean_volume_path,
            moving_img_path=template_path,
            output_path=template_pet_space_path,
            interpolator='genericLabel',
            is_inverse=True,
            transform_path=transformation_path
        )

        # Partial Volume Correction (PETPVC)
        print(":gear: STEP 5. [bold green]Compute 4D Mask (PET Space) for Partial Volume Correction (PVC)")
        pvc_dir = os.path.join(output_dir, 'PVC')
        os.makedirs(pvc_dir, exist_ok=True)
        refmask_pet_space_4d_path = os.path.join(pvc_dir, 'refmask_pet_space_4d.nii.gz')
        _ = petpvc_create_4d_mask(
            template_path=template_pet_space_path,
            template_name=template,
            reference_name=reference_region,
            mask_4d_out=refmask_pet_space_4d_path
        )

        # Partial Volume Correction (if applicable)
        print(":gear: STEP 6. [bold green]Running Partial Volume Correction (PVC)")
        pet_3d_pvc_volume_dir = os.path.join(output_dir, "pet_3d_volumes_pvc")
        os.makedirs(pet_3d_pvc_volume_dir, exist_ok=True)
        for pet_3d_volume in os.listdir(pet_3d_volumes_dir):
            pet_3d_volume_path = os.path.join(pet_3d_volumes_dir, pet_3d_volume)
            pet_3d_volume_pvc_path = os.path.join(pet_3d_pvc_volume_dir, pet_3d_volume)
            # Run Iterative Yang PVC Method
            run_petpvc_iterative_yang(
                pet_3d_volume_path=pet_3d_volume_path,
                pet_4d_mask_path=refmask_pet_space_4d_path,
                pet_3d_volume_pvc_path=pet_3d_volume_pvc_path,
                x="6.0",
                y="6.0",
                z="6.0"
            )

        # Re-compute 4D PET image from list of 3D volumes which are now in
        # RSA orientation and corrected for partial volume
        # NOTE: since we are utilizing C3D it was not possible to simply change
        # orientation of 4D image, we had to 1st split into 3D images and then
        # correct for orientation
        print(":gear: STEP 7. [bold green]Re-computing PET 4D Image in RSA Orientation")
        pet_4d_rsa_volume_path = os.path.join(output_dir, 'pet_4d_rsa.nii')
        _ = compute_4d_image(
            volume_dir=pet_3d_pvc_volume_dir,
            img_4d_out=pet_4d_rsa_volume_path
        )

        # Execute Simplified Reference Tissue Model (SRTM)
        print(":gear: STEP 8. [bold green]Execute Simplified Reference Tissue Model (SRTM)")
        srtm_results_dir = os.path.join(output_dir, 'SRTM_RESULTS')
        call_srtm(
            pet_4d_path=pet_4d_rsa_volume_path,
            reference_mask_path=template_pet_space_path,
            output_dir=srtm_results_dir,
            model=model
        )
        return 0
