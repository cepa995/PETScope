import os
import re
import json
import shutil
import nibabel as nib
import subprocess
import functools
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from typing import Any, Dict, Union, List
from nilearn.image import mean_img, concat_imgs
from scipy.signal import savgol_filter
from petscope.constants import REFERENCE_REGIONS, SETTINGS_JSON, SUPPORTED_PHYSICAL_SPACES, \
    SUPPORTED_REFERENCE_REGIONS
from typing import Dict, List

def check_if_reference_region_is_supported(reference_region: str) -> bool:
    """
    Checks if computation is supported for the specified reference region.

    Args:
        reference_region (str): The name of the reference region (e.g., "WholeCerebellum",
        "WholeWhiteMatter").

    Returns:
        bool: True if the specified reference region is supported, False otherwise.
    """
    return reference_region in SUPPORTED_REFERENCE_REGIONS

def check_if_physical_space_is_supported(physical_space: str) -> bool:
    """
    Checks if computation in the specified physical space is supported.

    Args:
        physical_space (str): The name of the physical space (e.g., "MRI", "PET").

    Returns:
        bool: True if the specified space is supported, False otherwise.
    """
    return physical_space in SUPPORTED_PHYSICAL_SPACES


def copy_file_to_directory(file_path, target_directory):
    """
    Copies a file to the specified target directory.

    Args:
        file_path (str): Absolute path of the file to be copied.
        target_directory (str): Directory where the file should be copied.

    Returns:
        str: Absolute path to the copied file in the target directory.

    Example:
        copy_file_to_directory("/path/to/file.nii", "/target/directory/")
    """
    os.makedirs(target_directory, exist_ok=True)
    target_path = os.path.join(target_directory, os.path.basename(file_path))
    shutil.copy(file_path, target_path)
    return target_path

def generate_docker_run_cmd(
    image_name: str,
    mount_points: Dict[str, str] = None,  # https://docs.docker.com/storage/volumes/
    volumes: Dict[str, str] = None,
    env_variables: Dict[str, str] = None,
    entrypoint: str = None,
    commands: List[str] = None,
    extra_parameters: List[str] = None,
    gpus: int = None,
    remove_container: bool = True
) -> List[str]:
    """
    Composes a Docker run command with the provided configuration options.

    Args:
        image_name (str): Name of the Docker image.
        mount_points (Dict[str, str], optional): Bind mount points in source:target format.
        volumes (Dict[str, str], optional): Docker volumes in source:target format.
        env_variables (Dict[str, str], optional): Environment variables as key:value pairs.
        entrypoint (str, optional): Custom entrypoint for the Docker container.
        commands (List[str], optional): Commands to execute within the container.
        extra_parameters (List[str], optional): Additional Docker parameters.
        gpus (int, optional): Number of GPUs to allocate for the container.
        remove_container (bool): Whether to remove the container after it exits. Default is True.

    Returns:
        List[str]: The composed Docker run command as a list of strings.

    Example:
        generate_docker_run_cmd(
            image_name="my_docker_image",
            mount_points={"/local/data": "/container/data"},
            env_variables={"MY_VAR": "value"},
            gpus=1,
            commands=["python", "script.py"]
        )
    """
    docker_command = ["docker", "run"]

    # Add mount points to the command if provided
    if mount_points:
        for src, dst in mount_points.items():
            docker_command.extend(["--mount", f"type=bind,source={src},target={dst}"])

    # Add volume mounts to the command if provided
    if volumes:
        for src, dst in volumes.items():
            docker_command.extend(["--mount", f"type=volume,source={src},target={dst}"])

    # Specify GPU allocation if provided
    if gpus:
        docker_command.extend(["--gpus", f"{gpus}"])

    # Add environment variables to the command if provided
    if env_variables:
        for key, value in env_variables.items():
            docker_command.extend(["-e", f"{key}={value}"])

    # Specify a custom entrypoint if provided
    if entrypoint:
        docker_command.extend(["--entrypoint", entrypoint])

    # Add the option to remove the container after it exits if set to True
    if remove_container:
        docker_command.extend(["--rm", image_name])
    else:
        docker_command.extend([image_name])

    # Add commands to run inside the container if provided
    if commands:
        docker_command.extend(commands)

    # Add any extra parameters provided to the command
    if extra_parameters:
        docker_command.extend(extra_parameters)

    return docker_command

def validate_settings_json(pet_image_path: str, settings_json: Dict[str, Any]) -> bool:
    """
    Validates the input settings JSON against a sample structure and checks PET image consistency.

    Args:
        pet_image_path (str): Absolute path to the PET image.
        settings_json (Dict[str, Any]): JSON-like dictionary of the PET settings.

    Returns:
        bool: True if the settings JSON is valid, False otherwise.

    Raises:
        SettingsJSONInvalidStructureException: If the JSON structure does not match the sample.
        PETImageNotFoundException: If the PET image path does not exist.
        PET3DImageException: If the PET image is not 4D.
        FrameNumberMismatchException: If the number of PET frames does not match the JSON configuration.
        FrameStartTimeAndOrDurationException: If frame start times and durations do not align.
        PETDataUnitsException: If the PET image data units are not in kBq/mL.

    Example:
        validate_settings_json("/path/to/pet_image.nii", settings_json)
    """
    # Sample JSON structure with required keys and corresponding value types
    sample_json = {
        "pet_json": {
            "AcquisitionMode": "4D",
            "AttenuationCorrection": "Activity decay corrected",
            "BodyPart": "brain",
            "FrameDuration": [
                15, 15, 15, 15, 30, 30, 30, 30, 60, 60, 60, 60, 60,
                180, 180, 180, 180, 300, 300, 300, 300, 300, 300, 300,
                300, 300, 300, 300
            ],
            "FrameTimesStart": [
                0, 15, 30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 420,
                480, 660, 840, 1020, 1200, 1500, 1800, 2100, 2400, 2700,
                3000, 3300, 3600, 3900
            ],
            "ImageDecayCorrected": "true",
            "ImageDecayCorrectionTime": "0",
            "InjectedMass": "5",
            "InjectedMassUnits": "ug",
            "InjectedRadioactivity": "185",
            "InjectedRadioactivityUnits": "MBq",
            "InjectionEnd": "30",
            "InjectionStart": 0,
            "Manufacturer": "Siemens",
            "ManufacturersModelName": "Biograph mMr",
            "ModeOfAdministration": "bolus",
            "ReconFilterSize": "2.5",
            "ReconFilterType": "PSF",
            "ReconMethodName": "MLEM",
            "ReconMethodParameterLabels": "iterations",
            "ReconMethodParameterUnits": "none",
            "ReconMethodParameterValues": "100",
            "ScanStart": 0,
            "SpecificRadioactivity": "35",
            "SpecificRadioactivityUnits": "GBq/ug",
            "TimeZero": "09:45:00",
            "TracerName": "SV2A",
            "TracerRadionuclide": "F18",
            "Units": "kBq/mL"
        }
    }

    # Helper function to check if two dictionaries match in structure and type
    def validate_dict_structure(reference: Dict[str, Any], test: Dict[str, Any]) -> bool:
        if not isinstance(test, dict):
            return False
        for key, value in reference.items():
            if key not in test:
                return False
            if isinstance(value, dict):
                if not validate_dict_structure(value, test[key]):
                    return False
            elif isinstance(value, list):
                if not isinstance(test[key], list):
                    return False
                if len(value) > 0 and not all(isinstance(elem, type(value[0])) for elem in test[key]):
                    return False
            else:
                if not isinstance(test[key], type(value)):
                    return False
        return True

    # Step 1. Validate dictionary structure
    is_structure_valid = validate_dict_structure(sample_json, settings_json)
    if not is_structure_valid:
        from petscope.exceptions import SettingsJSONInvalidStructureException
        raise SettingsJSONInvalidStructureException("Check data types and required\
                keys in settings_template.json file")
    
    # Step 2. Validate number of frames in PET image against number of frames
    # specified in FrameStart list of settings_template.json file
    frame_start_time_list = settings_json["pet_json"]["FrameTimesStart"]
    frame_duration_list = settings_json["pet_json"]["FrameDuration"]

    if not os.path.exists(pet_image_path):
        from petscope.exceptions import PETImageNotFoundException
        raise PETImageNotFoundException(f"Path to the PET image does not exist {pet_image_path}")
    
    # Load PET image for additional checks
    pet_image_nii = nib.load(pet_image_path)
    pet_image_data = pet_image_nii.get_fdata()
    if not len(pet_image_data.shape) == 4:
        from petscope.exceptions import PET3DImageException
        raise PET3DImageException("PET image should be 4D, got 3D instead")
    
    number_of_time_frames = pet_image_data.shape[3]
    frame_start_time_list_num = len(frame_start_time_list)
    if number_of_time_frames != frame_start_time_list_num:
        from petscope.exceptions import FrameNumberMismatchException
        raise FrameNumberMismatchException(f"Found {number_of_time_frames} time frames in PET 4D "
               + f"while in settings JSON found {frame_start_time_list_num}")
    
    # Step 3. Validate Frame Start Time against Frame Duration
    for i in range(0, len(frame_start_time_list)-1):
        if frame_duration_list[i] + frame_start_time_list[i] != frame_start_time_list[i+1]:
            from petscope.exceptions import FrameStartTimeAndOrDurationException
            raise FrameStartTimeAndOrDurationException("There is a disagreement between "
                    + "frame start time and frame duration lists in settings_template.json")
        
    # Step 4. Make sure PET data is in kBq/mL units, otherwise we need scanner calibration
    # factor to normalize voxel intensities and convert to kBq/mL units, which are needed
    # for TAC computation
    if settings_json["pet_json"]["Units"].lower() != "kbq/ml":
        from petscope.exceptions import PETDataUnitsException
        raise PETDataUnitsException(f"Expected kBq/mL units, got {settings_json['pet_json']['Units']} instead")
    return True

def read_settings_json(pet_image_path: str) -> Dict[str, Union[int, str, List[str]]]:
    """
    Reads and validates the settings JSON file for PET analysis.

    Args:
        pet_image_path (str): Absolute path to the PET image.

    Returns:
        Dict[str, Union[int, str, List[str]]]: A validated dictionary of settings for PET analysis.

    Raises:
        SettingsJSONTemplateNotFoundException: If the settings JSON template is not found.
        InvalidSettingsJSONTemplateFileException: If the settings JSON file is invalid.

    Example:
        settings = read_settings_json("/path/to/pet_image.nii")
    """
    # Check if the path to settings JSON template exists
    if not os.path.exists(SETTINGS_JSON):
        from petscope.exceptions import SettingsJSONTemplateNotFoundException
        raise SettingsJSONTemplateNotFoundException(f"JSON Settings Template "
               + f"was not found at {SETTINGS_JSON}")
    
    # Read settings JSON template
    settings_json_file = open(SETTINGS_JSON)
    settings_json = json.load(settings_json_file)

    # Validate settings JSON template to make sure user did not remove or 
    # ommit any of the required keys
    if not validate_settings_json(pet_image_path, settings_json):
        from petscope.exceptions import InvalidSettingsJSONTemplateFileException
        raise InvalidSettingsJSONTemplateFileException("Settings JSON File {SETTINGS_jSON}\
                is invalid. Please double check its content")

    # Return PET related settings
    return settings_json["pet_json"]


def get_reference_region_mask(
        template_path: str,
        template_name: str,
        reference_name: str,
        mask_out: str
    ) -> nib.Nifti1Image:
    """
    Creates a 3D reference region mask from a template.

    Args:
        template_path (str): Absolute path to the template mask.
        template_name (str): Name of the template (e.g., "FreeSurfer").
        reference_name (str): Name of the desired reference region (e.g., "WholeCerebellum").
        mask_out (str): Absolute path where the resulting 3D mask will be saved.

    Returns:
        nib.Nifti1Image: The created 3D reference region mask as a NIfTI image.

    Example:
        get_reference_region_mask(
            "/path/to/template.nii",
            "FreeSurfer",
            "WholeCerebellum",
            "/output/mask.nii"
        )
    """
    reference_region_labels = REFERENCE_REGIONS[template_name][reference_name]
    # Create output directory if it doesn't exist
    dirname = os.path.dirname(mask_out)
    os.makedirs(dirname, exist_ok=True)

    # Load the 3D mask data
    mask_3d_nii = nib.load(template_path)
    mask_3d_data = mask_3d_nii.get_fdata().astype(np.uint16)

    # Create a binary mask that combines all labels
    mask = functools.reduce(np.logical_or, (mask_3d_data == lbl for lbl in reference_region_labels))
    masked_image = np.where(mask, mask_3d_data, 0)
    masked_image[masked_image != 0] = 1
    masked_image_nii = nib.Nifti1Image(masked_image.astype(np.uint8), mask_3d_nii.affine)
    nib.save(masked_image_nii, mask_out)

    # Return Nifti image as a result
    return masked_image_nii

def compute_time_activity_curve(
        pet_image_path: str,
        template_path: str,
        template_name: str,
        reference_name: str,
        time_activity_curve_out: str,
        frame_start_times: List[int],
        frame_durations: List[int],
        window_length: int = None,
        polyorder: int = None,
        debug: bool = False
    ) -> np.array:
    """
    Computes a Time Activity Curve (TAC) for a given reference region from a PET image.

    This function calculates the average activity in the specified reference region for 
    each time frame in the PET image. Optionally, it applies Savitzky-Golay smoothing to 
    the TAC and generates a plot saved to the specified output path.

    Args:
        pet_image_path (str): Absolute path to the input 4D PET image.
        template_path (str): Absolute path to the template mask image.
        template_name (str): Name of the template being used (e.g., "FreeSurfer").
        reference_name (str): Name of the reference region (e.g., "WholeCerebellum").
        time_activity_curve_out (str): Absolute path to save the TAC plot as an image file.
        frame_start_times (List[int]): List of start times for each PET time frame (in seconds).
        frame_durations (List[int]): List of durations for each PET time frame (in seconds).
        window_length (int, optional): Window size for Savitzky-Golay smoothing. Defaults to None.
        polyorder (int, optional): Polynomial order for Savitzky-Golay smoothing. Defaults to None.
        debug (bool, optional): If True, saves intermediate masked PET frames for debugging. Defaults to False.

    Returns:
        np.array: Computed Time Activity Curve. If smoothing is applied, the smoothed TAC is returned; otherwise, 
                  the original TAC is returned.

    Raises:
        SavitzkyGolaySmoothingException: If `window_length` is smaller than `polyorder`.

    Notes:
        - The reference region mask is derived from the template and the specified reference name.
        - If smoothing parameters are provided, the TAC is smoothed using the Savitzky-Golay filter.
        - Debug mode saves intermediate masked images for each time frame to the output directory.

    Example:
        compute_time_activity_curve(
            pet_image_path="path/to/pet_image.nii",
            template_path="path/to/template.nii",
            template_name="FreeSurfer",
            reference_name="WholeCerebellum",
            time_activity_curve_out="path/to/output/tac.png",
            frame_start_times=[0, 60, 120],
            frame_durations=[60, 60, 60],
            window_length=5,
            polyorder=3,
            debug=True
        )
    """
    # Create directory if it does not exist
    dirname = os.path.dirname(time_activity_curve_out)
    os.makedirs(dirname, exist_ok=True)

    # Get Reference Region/Mask
    reference_mask_path = os.path.join(dirname, "reference_mask.nii")
    reference_region_img_nii = get_reference_region_mask(
        template_path=template_path,
        template_name=template_name,
        reference_name=reference_name,
        mask_out=reference_mask_path
    )
    reference_region_img_data = reference_region_img_nii.get_fdata().astype(np.uint8)

    # Compute frame midpoint times
    frame_start_times = np.array(frame_start_times)
    frame_durations = np.array(frame_durations)
    frame_midpoint_times = frame_start_times + frame_durations / 2

    # Load PET image
    pet_img_nii = nib.load(pet_image_path)
    pet_img_data = pet_img_nii.get_fdata()

    # Calculate the TAC by averaging over the ROI for each time frame
    tac = []
    for t in range(pet_img_data.shape[3]): # Looping over time-frames
        pet_time_frame = pet_img_data[:, :, :, t]
        pet_data_masked = np.multiply(pet_time_frame, reference_region_img_data)
        if debug:
            pet_data_masked_nii = nib.Nifti1Image(pet_data_masked, pet_img_nii.affine)
            nib.save(pet_data_masked_nii, os.path.join(dirname, f"pet_data_masked_{t}.nii"))
        average_activity = np.mean(pet_data_masked)
        print(f"\t:chart_increasing: [bold green]Average Time Activity for Frame [/]{t} [bold green]is [/]{average_activity}")
        tac.append(average_activity)
    
    # Convert TAC to numpy array for easier handling
    tac = np.array(tac)

    # Plot the Time Activity Curve and save it
    plt.figure(figsize=(10, 5))
    plt.plot(frame_midpoint_times, tac, marker='o', color='blue', label='Original TAC')

    # TAC smoothing
    smoothed_tac = None
    if window_length and polyorder:
        if window_length < polyorder:
            from petscope.exceptions import SavitzkyGolaySmoothingException
            raise SavitzkyGolaySmoothingException(f"Windows size ({window_length})\
                    cannot be smaller then polynomial order ({polyorder})")
        print(f"\tSavitzky Golay Smoothing with Window Size = {window_length} and Polynomial Order = {polyorder}")
        smoothed_tac = savgol_filter(tac, window_length, polyorder)
        plt.plot(frame_midpoint_times, smoothed_tac, marker='x', color='red', label='Smoothed TAC', linewidth=2)

    plt.title('Time Activity Curve (TAC)')
    plt.xlabel('Time')
    plt.ylabel('Average Activity (kBq/mL)')
    plt.grid()

    # Save the figure
    plt.savefig(time_activity_curve_out, dpi=300, bbox_inches='tight')  
    plt.close()  
    
    # Return Time Activity Curve
    if smoothed_tac is not None:
        return smoothed_tac
    return tac

def change_dtype(image_path, output_path, dtype):
    """
    Changes the data type of an image.

    Args:
        image_path (str): Absolute path to the input image.
        output_path (str): Absolute path to save the output image.
        dtype (str): Target data type (e.g., "char", "float", "int").

    Example:
        change_dtype("/path/to/image.nii", "/output/image.nii", "float")
    """
    subprocess.run([
        'c3d',
        image_path,
        '-type', dtype,
        "-o", output_path
    ])


def change_orientation(image_path, output_path, orientation_code='RSA'):
    """
    Changes the orientation of an image according to the specified code.

    Args:
        image_path (str): Absolute path to the input image.
        output_path (str): Absolute path to save the output image.
        orientation_code (str): Orientation code (e.g., "RSA", "LPI").

    Example:
        change_orientation("/path/to/image.nii", "/output/image.nii", "LPI")
    """
    subprocess.run([
        'c3d',
        image_path,
        '-swapdim', orientation_code,
        "-o", output_path
    ])


def compute_4d_image(volume_dir, img_4d_out):
    """
    Creates a 4D image from a directory of 3D volumes.

    Args:
        volume_dir (str): Directory containing 3D volume images.
        img_4d_out (str): Absolute path to save the output 4D image.

    Returns:
        nib.Nifti1Image: The created 4D image.

    Example:
        compute_4d_image("/path/to/3d_volumes", "/output/4d_image.nii")
    """
    volumes_nii = [nib.load(os.path.join(volume_dir, f)) for f in os.listdir(volume_dir)]
    img_4d_nii = concat_imgs(volumes_nii)
    nib.save(img_4d_nii, img_4d_out)
    return img_4d_nii


def compute_mean_volume(volume_dir, mean_3d_out):
    """
    Computes the mean 3D volume from a list of 3D volumes.

    Args:
        volume_dir (str): Directory containing 3D volume images.
        mean_3d_out (str): Absolute path to save the mean 3D volume.

    Returns:
        nib.Nifti1Image: The computed mean 3D image.

    Example:
        compute_mean_volume("/path/to/3d_volumes", "/output/mean_image.nii")
    """
    volumes_nii = [nib.load(os.path.join(volume_dir, f)) for f in os.listdir(volume_dir)]
    mean_3d_image = mean_img(volumes_nii, volumes_nii[0].affine)
    nib.save(mean_3d_image, mean_3d_out)
    return mean_3d_image


def compute_3D_volume(nifti_4d_path, output_file):
    """
    Converts a 4D NIfTI image into a single 3D volume by averaging over time frames.

    Args:
        nifti_4d_path (str): Absolute path to the input 4D NIfTI image.
        output_file (str): Absolute path to save the output 3D volume.

    Returns:
        nib.Nifti1Image: The resulting 3D image.

    Example:
        compute_3D_volume("/path/to/4d_image.nii", "/output/3d_image.nii")
    """   
    nifti_4d = nib.load(nifti_4d_path)
    nifti_4d_data = nifti_4d.get_fdata()
    
    # Check if the image has 4 dimensions
    assert len(nifti_4d_data.shape) == 4, f"{nifti_4d_path} image must have exactly 4 dimensions"
    
    # Check if the image is not empty
    assert nifti_4d_data.min() != nifti_4d_data.max(), f"{nifti_4d_path} image appears to be empty"
    
    # Convert to 3D volume (Expected Shape X, Y, Z, C)
    rank = 0.20
    num_frames = nifti_4d_data.shape[3]
    first_frame = int(np.floor(num_frames * rank))
    last_frame = num_frames
    volume_subset = nifti_4d_data[:, :, :, first_frame:last_frame]
    volume_src = np.mean(volume_subset, axis=3)
    
    output_3d_img = nib.Nifti1Image(volume_src, nifti_4d.affine)
    nib.save(output_3d_img, output_file)
    return output_3d_img


def convert_4d_to_3d(img_4d_path, img_3d_dir, prefix='pet_3d_', orientation=None):
    """
    Converts a 4D image into a sequence of 3D volumes.

    Args:
        img_4d_path (str): Absolute path to the 4D image.
        img_3d_dir (str): Directory to save the 3D volumes.
        prefix (str, optional): Prefix for the 3D volume filenames. Defaults to 'pet_3d_'.
        orientation (str, optional): Orientation code to apply to each volume.

    Example:
        convert_4d_to_3d("/path/to/4d_image.nii", "/output/3d_volumes")
    """
    os.makedirs(img_3d_dir, exist_ok=True)
    img_4d_nii = nib.load(img_4d_path)
    img_4d_data = img_4d_nii.get_fdata()
    num_3d_imgs = img_4d_data.shape[-1]
    
    for idx in range(num_3d_imgs):
        img_3d_vol_data = img_4d_data[..., idx]
        img_3d_vol_nii = nib.Nifti1Image(img_3d_vol_data, img_4d_nii.affine)
        img_3d_vol_out = os.path.join(img_3d_dir, f'{prefix}_{idx}.nii')
        nib.save(img_3d_vol_nii, img_3d_vol_out)
        
        if orientation is not None:
            # Split 4D image into sequence of 3D volumes and change orientation for each
            change_orientation(
                image_path=img_3d_vol_out,
                orientation_code=orientation,
                output_path=img_3d_vol_out
            )

def get_orientation(nifti_image_path) -> str:
    """
    Determines the orientation of a NIfTI image.

    Args:
        nifti_image_path (str): Absolute path to the NIfTI image.

    Returns:
        str: Orientation code (e.g., "RSA", "LPI").

    Example:
        orientation = get_orientation("/path/to/image.nii")
    """
    image_nii = nib.load(nifti_image_path)
    x, y, z = nib.aff2axcodes(image_nii.affine)
    return x + y + z

def extract_image_info(image_path):
    """
    Extracts dimension, bounding box, and orientation information from a NIfTI image.

    Args:
        image_path (str): Absolute path to the NIfTI image.

    Returns:
        Tuple[List[float], List[List[float]], str]: Dimensions, bounding box, and orientation.

    Example:
        dim, bb, orient = extract_image_info("/path/to/image.nii")
    """
    c3d_info_cmd = ["c3d", image_path, "-info"]
    result = subprocess.run(c3d_info_cmd, capture_output=True, text=True)
    info = result.stdout

    # Regular expressions to find 'dim', 'bb', and 'orient' values
    dim_match = re.search(r'dim = \[([^\]]+)\]', info)
    bb_match = re.search(r'bb = \{([^\}]+)\}', info)
    orient_match = re.search(r'orient = (\w+)', info)

    # Extract and convert values if matches are found
    dim = [float(x) for x in dim_match.group(1).split(',')] if dim_match else None
    bb = [[float(x) for x in point.replace('[', '').replace(']', '').split()] for point in bb_match.group(1).split('], [')] if bb_match else None
    orient = orient_match.group(1) if orient_match else None

    return dim, bb, orient

def c3d_space_check(image1_path, image2_path) -> bool:
    """
    Checks whether two images share the same space using C3D.

    Args:
        image1_path (str): Absolute path to the first image.
        image2_path (str): Absolute path to the second image.

    Returns:
        bool: True if the images share the same space, False otherwise.

    Example:
        same_space = c3d_space_check("/path/to/image1.nii", "/path/to/image2.nii")
    """
    dim1, bb1, orient1 = extract_image_info(image1_path)
    dim2, bb2, orient2 = extract_image_info(image2_path)

    return dim1 == dim2 and bb1 == bb2 and orient1 == orient2

def c3d_copy_transform(src, dst) -> None:
    """
    Copies the image transform (header) from a source image to a destination image using C3D.

    Args:
        src (str): Absolute path to the source image.
        dst (str): Absolute path to the destination image.

    Example:
        c3d_copy_transform("/path/to/source.nii", "/path/to/destination.nii")
    """
    subprocess.run([
        'c3d',
        src,
        dst,
        '-copy-transform',
        "-o", dst
    ])