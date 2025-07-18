import os
import numpy as np
import nibabel as nib
from rich import print
from petscope.constants import TARGET_REGIONS
from typing import Dict, Any, List, Callable
from petscope.constants import PVC_SUPPORTED_METHODS, MRI_PHYSICAL_SPACE, SUPPORTED_PHYSICAL_SPACES, \
      SUPPORTED_REFERENCE_REGIONS, CUSTOM_PIPELINE_COMMAND_DICT
from petscope.dynamicpet_wrapper.srtm import call_srtm
from petscope.registration import ants_registration, ants_warp_image
from petscope.utils import compute_time_activity_curve, convert_4d_to_3d,\
      compute_mean_volume, compute_4d_image, c3d_space_check, check_if_physical_space_is_supported, \
      check_if_reference_region_is_supported, check_if_model_is_supported
from petscope.petpvc_wrapper.utils import petpvc_create_4d_mask, check_if_pvc_method_is_supported
from petscope.petpvc_wrapper.petpvc import run_petpvc_iterative_yang
from petscope.spm_wrapper.spm import spm_realignment, PET_REALIGN
from petscope.kinetic_modeling.dvr import compute_DVR_image

def map_strings_to_functions(strings: List[str], function_mapping: Dict[str, Callable]) -> Dict[str, Callable]:
    """
    Maps each string in a given list to a function based on a predefined dictionary.

    Args:
        strings (List[str]): A list of strings to map to functions.
        function_mapping (Dict[str, Callable]): A dictionary where keys are strings and
                                                values are function pointers.
    Returns:
        Dict[str, Callable]: A dictionary mapping input strings to their corresponding functions.
                             If a string has no matching function, it's not included.
    """
    mapped_functions = {}
    for string in strings:
        if string in function_mapping:
            mapped_functions[string] = function_mapping[string]
        else:
            print(f"Warning: No function found for '{string}'")
    return mapped_functions

def interactive_menu(menu: dict) -> List[str]:
    """
    Displays a hierarchical menu to configure a custom pipeline interactively.

    Users can navigate through menus and submenus, selecting actions to include in the pipeline.
    The user can view and confirm the configured pipeline at the end.

    Args:
        menu (dict): A nested dictionary representing the menu hierarchy. Keys are menu items,
                     and values are either strings (actions) or dictionaries (submenus).

    Returns:
        List[str]: A list of selected commands in the order they were chosen by the user.

    Menu Structure:
        - Coregister PET to MRI (and vice versa)
        - PET Specific Commands:
            - SPM Realignment
            - Partial Volume Correction (PVC)
            - Compute PET Mean Image
            - Compute Time Activity Curve (TAC)
        - MRI Specific Commands (currently not implemented)
        
    Example Usage:
        command_list = interactive_menu(menu_structure)
        # `command_list` contains selected commands
    """
    def display_menu(menu: dict, command_list: List[str] = [], path: str = "List of supported commands") -> List[str]:
        while True:
            print(f"\n{path}")
            print("-" * len(path))
            for idx, key in enumerate(menu.keys(), start=1):
                if key not in command_list:
                    print(f"{idx}. {key}")
            print("0. :rewind: [green]Go Back ")

            try:
                choice = int(input("Enter your choice (number): "))
                if choice == 0:
                    break
                elif 1 <= choice <= len(menu):
                    selected_key = list(menu.keys())[choice - 1]
                    selected_value = menu[selected_key]
                    
                    if isinstance(selected_value, dict):  # Navigate to submenu
                        display_menu(selected_value, command_list=command_list, path=f"{path} > {selected_key}")
                    else:
                        if selected_key in command_list:
                            print(f'[yellow] Command "{selected_key}" is already selected. Please choose another option.')
                        else:
                            print(f"[green] Action selected: :heavy_plus_sign: {selected_value}")
                            command_list.append(selected_key)
                else:
                    print(":cross_mark: Invalid choice. Please select a valid menu item.")
            except ValueError:
                print(":cross_mark: Invalid input. Please enter a number.")
        return command_list

    # Display the top-level menu
    command_list = display_menu(menu)
    if not command_list:
        return []

    print("\n:eyes: Pipeline Overview:")
    print("-" * 17)
    for idx, cmd in enumerate(command_list, start=1):
        print(f"{idx}. {cmd}")

    # Confirm the pipeline configuration
    answer = ''
    while answer.lower() not in ['yes', 'no']:
        answer = input("\nWrite pipeline configuration to disk? [Yes/No]: ")

    if answer.lower() == 'yes':
        print("\n:white_heavy_check_mark: Pipeline configuration saved to disk!")
    else:
        print("\n:cross_mark: Pipeline configuration discarded.")
    return command_list

class PETScope:
    def __init__(self) -> None:
        """
        Initializes the PETScope class.

        This class provides methods to process PET images, including:
        - Generating time-activity curves (TACs)
        - Registering PET images to T1 images
        - Performing partial volume correction (PVC)
        - Running the Simplified Reference Tissue Model (SRTM) pipeline
        """
        pass

    def custom_pipeline(self) -> None:
        """
        Runs a custom pipeline configured interactively through a hierarchical menu system.

        This command enables users to build a custom pipeline by selecting from a menu of options.
        Submenus allow for a structured selection of specific commands or actions to include in the pipeline.
        Once the user completes their selection, the pipeline configuration can be saved or discarded.

        Returns:
            None
        """
        # Call the interactive menu to configure the pipeline
        pipeline = interactive_menu(CUSTOM_PIPELINE_COMMAND_DICT)
        for step in pipeline:
            pass
        
    def coregister_pet_and_mr(
        self,
        pet_path: str,
        t1_3d_path: str,
        type_of_transform: str,
        output_dir: str
    ) -> int:
        """
        Registers a PET image to a T1 image using ANTs.

        This method splits the 4D PET image into 3D volumes, computes the mean PET volume,
        and registers the mean PET image to the T1 image using the specified transformation type.

        Args:
            pet_path (str): Absolute path to the input 3D or 4D PET image.
            t1_3d_path (str): Absolute path to the input T1 image.
            type_of_transform (str): Type of transformation to perform (e.g., "Rigid", "Affine").
            output_dir (str): Directory to save the registration results.

        Returns:
            int: Returns 0 upon successful completion.

        Example:
            coregister_pet_and_mr((
                pet_path="/path/to/pet_4d.nii",
                t1_3d_path="/path/to/t1.nii",
                type_of_transform="Rigid",
                output_dir="/path/to/output"
            )
        """
        pet_img_dims = nib.load(pet_path).get_fdata().shape
        if len(pet_img_dims) == 4:
            # Convert 4D PET image to sequence of 3D volumes
            print(":gear: STEP 1. [bold green]Converting 4D PET to Sequence of 3D Volumes")
            pet_3d_volumes_dir = os.path.join(output_dir, "pet_3d_volumes")
            os.makedirs(pet_3d_volumes_dir, exist_ok=True)
            convert_4d_to_3d(
                img_4d_path=pet_path,
                img_3d_dir=pet_3d_volumes_dir,
                orientation='RSA'
            )

            # Compute PET 3D Mean Volume
            print(":gear: STEP 2. [bold green]Computing MEAN 3D Volume")
            pet_path = os.path.join(output_dir, 'pet_3d_mean.nii')
            _ = compute_mean_volume(
                volume_dir=pet_3d_volumes_dir,
                mean_3d_out=pet_path
            )

        # Rigid Registration - PET to T1 Space
        print(f":gear: STEP 3. [bold green]Running ANTs {type_of_transform} Registration")
        registration_dir = os.path.join(output_dir, 'pet_to_t1_registration')
        os.makedirs(registration_dir, exist_ok=True)
        _ = ants_registration(
            moving_img_path=pet_path,
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
        physical_space: str,
        reference_region: str,
        target_region: str,
        output_dir: str,
        model: str,
        pvc_method: str,
        window_size: int,
        polynomial_order: int,
        pet_json: Dict[str, Any]
    ):
        """
        Executes the Simplified Reference Tissue Model (SRTM) pipeline on PET data.

        This pipeline includes:
        - Validating input arguments
        - Realigning PET images
        - Registering PET images to T1 images
        - Performing partial volume correction (PVC)
        - Computing the Time Activity Curve (TAC)
        - Running the SRTM model

        Args:
            pet_4d_path (str): Absolute path to the input 4D PET image.
            t1_3d_path (str): Absolute path to the input T1 MRI image.
            template_path (str): Absolute path to the reference template image.
            template (str): Name of the reference template.
            physical_space (str): Target physical space for the output (e.g., "MRI_PHYSICAL_SPACE").
            reference_region (str): Name of the reference region for TAC computation (e.g., "WholeCerebellum").
            target_region (str): Name of the target region for TAC computation (e.g., "Hippocampus").
            output_dir (str): Directory to save all pipeline results.
            model (str): SRTM model type to use.
            pvc_method (str): Partial volume correction method (e.g., "Iterative Yang").
            window_size (int): Length of the window for Savitzky-Golay smoothing during TAC computation.
            polynomial_order (int): Polynomial order for Savitzky-Golay smoothing during TAC computation.
            pet_json (Dict[str, Any]): Dictionary containing PET metadata (e.g., frame times, durations).

        Returns:
            int: Returns 0 upon successful completion.

        Raises:
            NotSamePhysicalSpaceException: If the T1 and template images are not in the same space.
            PVCMethodSupportException: If the specified PVC method is not supported.
            PhysicalSpaceSupportException: If the specified physical space is not supported.
        """
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
        # Check if physical space passed as an argument is supported
        if physical_space and not check_if_physical_space_is_supported(physical_space):
            from petscope.exceptions import PhysicalSpaceSupportException
            raise PhysicalSpaceSupportException(f"Computation is not supported in {physical_space} space " + 
                                            f" . Please choose one of the following {SUPPORTED_PHYSICAL_SPACES}")
        # Check if reference region  passed as an argument is supported
        if reference_region and not check_if_reference_region_is_supported(reference_region):
            from petscope.exceptions import ReferenceRegionSupportException
            raise ReferenceRegionSupportException(f"Specified reference region - {reference_region} is NOT supported " + 
                                            f" . Please choose one of the following {SUPPORTED_REFERENCE_REGIONS}")
        # Check if model argument is specified, in that case we rely on dynamicpet package (https://github.com/bilgelm/dynamicpet)
        if model and not check_if_model_is_supported(model):
            from petscope.exceptions import DynmicPetWrapperException
            raise DynmicPetWrapperException(f"Model {model} is not supported by dynamicpet. Please choose" + 
                                            f" one of the following models {SUPPORTED_DYNAMICPET_MODELS}")
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

        # In case of MRI Phyiscal space, warp all 3D PET volumes to MRI space
        if physical_space == MRI_PHYSICAL_SPACE:
            volume_dir = pet_3d_pvc_volume_dir if pvc_method else pet_3d_volumes_dir
            for volume_3d in os.listdir(volume_dir):
                volume_path = os.path.join(volume_dir, volume_3d)
                # Warp PET 3D Volume to MR space
                ants_warp_image(
                    fixed_img_path=t1_3d_path,
                    moving_img_path=volume_path,
                    output_path=volume_path,
                    interpolator='linear',
                    transform_path=transformation_path
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
            template_path=template_pet_space_path if physical_space and physical_space != MRI_PHYSICAL_SPACE else template_path,
            template_name=template,
            reference_name=reference_region,
            time_activity_curve_out=tac_out,
            window_length=window_size,
            polyorder=polynomial_order,
            frame_start_times=pet_json["FrameTimesStart"],
            frame_durations=pet_json["FrameDuration"]
        )

        # Define list of target regions (if not specified, take all the ones specified within the FreeSurfer)
        srtm_results_dir = os.path.join(output_dir, 'SRTM_RESULTS')
        target_regions = [target_region] if target_region else [region for region in TARGET_REGIONS['FreeSurfer'] if region != reference_region]
        if not model:
            # Execute Simplified Reference Tissue Model2 (SRTM2)
            compute_DVR_image(
                pet_file=pet_4d_rsa_volume_path,
                frame_durations=np.array(pet_json['FrameDuration']),
                output_dir=srtm_results_dir,
                template_path=template_pet_space_path if physical_space and physical_space != MRI_PHYSICAL_SPACE else template_path,
                template_name=template,
                roi_regions= target_regions,
                reference_region=reference_region
            )
        else:
            # Perform SRTM with dynamicpet
            call_srtm(
                pet_4d_path=pet_4d_rsa_volume_path,
                template_name=template,
                template_path=template_pet_space_path if physical_space and physical_space != MRI_PHYSICAL_SPACE else template_path,
                reference_region=reference_region,
                model=model,
                target_regions=target_regions,
                output_dir=srtm_results_dir
            )

        return 0