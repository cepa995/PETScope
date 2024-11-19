import os
import numpy as np
from rich import print
from typing import Dict, Any
from petscope.constants import PVC_SUPPORTED_METHODS
from petscope.dynamicpet_wrapper.srtm import call_srtm
from petscope.registration import ants_registration, ants_warp_image
from petscope.utils import compute_time_activity_curve, convert_4d_to_3d,\
      compute_mean_volume, compute_4d_image, c3d_space_check
from petscope.petpvc_wrapper.utils import petpvc_create_4d_mask, check_if_pvc_method_is_supported
from petscope.petpvc_wrapper.petpvc import run_petpvc_iterative_yang
from petscope.spm_wrapper.spm import spm_realignment, PET_REALIGN

class PETScope:
    def __init__(self) -> None:
        pass

    def get_tac(
        self,
        pet_image_path: str,
        template_path: str,
        template_name: str,
        reference_name: str,
        time_activity_curve_out: str,
        window_length: int = None,
        polyorder: int = None
    ) -> np.array:
        _, _ = compute_time_activity_curve(
            pet_image_path=pet_image_path,
            template_path=template_path,
            template_name=template_name,
            reference_name=reference_name,
            time_activity_curve_out=time_activity_curve_out,
            window_length=window_length,
            polyorder=polyorder
        )

        return 0

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
        model: str,
        pvc_method: str,
        window_size: int,
        polynomial_order: int,
        pet_json: Dict[str, Any]
    ):
        print(":gear: STEP 0. [bold green]Validating Input Arguments")
        # Check if T1 image and Template image are in the same space
        if not c3d_space_check(template_path, t1_3d_path):
            from petscope.exceptions import NotSamePhysicalSpaceException
            raise NotSamePhysicalSpaceException(
                f"Template image {template_path} is not in the same space as T1 image {t1_3d_path}"
            )
        # Check if PVC method passed as an argument is supported
        if pvc_method and not check_if_pvc_method_is_supported(pvc_method):
            from petscope.exceptions import PVCMethodSupportException
            raise PVCMethodSupportException(f"PVC Method {pvc_method} is not supported! Please choose from " + 
                                            f"{PVC_SUPPORTED_METHODS}")
        print("\t:white_heavy_check_mark: [bold green]INPUTS ARE VALID!")
        
        # Realignment via SPM
        print(f":gear: STEP 1. [bold green]Executing SPM Realignment for {os.path.basename(pet_4d_path)}")
        mounting_point_dir = os.path.join(output_dir, "realignment_dir")
        realignment_results = spm_realignment(
            realignment_out_dir=mounting_point_dir,
            pet_4d_path=pet_4d_path
        )
        # Update PET 4D Path to a newly realigned image
        print(f":zap: [bold green]Updating PET 4D path to {realignment_results[PET_REALIGN]} (SPM Realignment)")
        pet_4d_path = realignment_results[PET_REALIGN]

        # Convert 4D PET image to sequence of 3D volumes
        print(":gear: STEP 2. [bold green]Converting 4D PET to Sequence of 3D Volumes")
        pet_3d_volumes_dir = os.path.join(output_dir, "pet_3d_volumes")
        os.makedirs(pet_3d_volumes_dir, exist_ok=True)
        convert_4d_to_3d(
            img_4d_path=pet_4d_path,
            img_3d_dir=pet_3d_volumes_dir,
            orientation='RSA'
        )

        # Compute PET 3D Mean Volume
        print(":gear: STEP 3. [bold green]Computing MEAN 3D Volume")
        pet_3d_mean_volume_path = os.path.join(output_dir, 'pet_3d_mean.nii')
        _ = compute_mean_volume(
            volume_dir=pet_3d_volumes_dir,
            mean_3d_out=pet_3d_mean_volume_path
        )

        # Rigid Registration - PET to T1 Space
        print(":gear: STEP 4. [bold green]Running ANTs Rigid Registration")
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
        print(":gear: STEP 5. [bold green]Warp Reference Mask (MNI) to PET Space")
        template_pet_space_path = os.path.join(registration_dir, 'template_pet_space.nii')
        ants_warp_image(
            fixed_img_path=pet_3d_mean_volume_path,
            moving_img_path=template_path,
            output_path=template_pet_space_path,
            interpolator='genericLabel',
            is_inverse=True,
            transform_path=transformation_path
        )

        if pvc_method:
            # Partial Volume Correction (PETPVC)
            print(":gear: STEP 6. [bold green]Compute 4D Mask (PET Space) for Partial Volume Correction (PVC)")
            pvc_dir = os.path.join(output_dir, 'PVC')
            os.makedirs(pvc_dir, exist_ok=True)
            refmask_pet_space_4d_path = os.path.join(pvc_dir, 'refmask_pet_space_4d.nii.gz')
            _ = petpvc_create_4d_mask(
                template_path=template_pet_space_path,
                template_name=template,
                reference_name=reference_region,
                mask_4d_out=refmask_pet_space_4d_path
            )

            print(":gear: STEP 7. [bold green]Running Partial Volume Correction (PVC)")
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
        print(":gear: STEP 8. [bold green]Re-computing PET 4D Image in RSA Orientation")
        pet_4d_rsa_volume_path = os.path.join(output_dir, 'pet_4d_rsa.nii')
        _ = compute_4d_image(
            volume_dir=pet_3d_pvc_volume_dir if pvc_method else pet_3d_volumes_dir,
            img_4d_out=pet_4d_rsa_volume_path
        )

        # Compute Time Activity Curve (TAC)
        print(":gear: STEP 9. [bold green]Computing Time Activity Curve (TAC)")
        tac_out = os.path.join(output_dir, 'time_activity_curve.png')
        compute_time_activity_curve(
            pet_image_path=pet_4d_rsa_volume_path,
            template_path=template_pet_space_path,
            template_name=template,
            reference_name=reference_region,
            time_activity_curve_out=tac_out,
            window_length=window_size,
            polyorder=polynomial_order,
            frame_start_times=pet_json["FrameTimesStart"],
            frame_durations=pet_json["FrameDuration"]
        )

        # Execute Simplified Reference Tissue Model (SRTM)
        print(":gear: STEP 10. [bold green]Execute Simplified Reference Tissue Model (SRTM)")
        srtm_results_dir = os.path.join(output_dir, 'SRTM_RESULTS')
        call_srtm(
            pet_4d_path=pet_4d_rsa_volume_path,
            reference_mask_path=template_pet_space_path,
            output_dir=srtm_results_dir,
            model=model
        )
        return 0
