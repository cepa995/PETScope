import pytest
import copy
import nibabel as nib
import numpy as np
from pathlib import Path

# Locate the PETScope-Test-Data directory relative to this file
TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "PETScope-Test-Data"

@pytest.fixture
def srtm2_test_args():
    """Arguments for srtm2 test"""
    # Define paths to required files
    moving_img_path = TEST_DATA_DIR / "SRTM2" / "CSV" / "simRef_example.csv"
    return {
        "file_path": str(moving_img_path)
    }

@pytest.fixture
def ants_registration_test_args():
    """Arguments for ants_registration test."""
    # Define paths to required files
    moving_img_path = TEST_DATA_DIR / "T1" / "Input" / "t1_mni.nii"
    fixed_img_path = TEST_DATA_DIR / "PET" / "Input" / "pet_3d.nii"
    registration_dir = TEST_DATA_DIR / "tests" / "reg_test_results"

    return {
        "moving_img_path": str(moving_img_path),
        "fixed_img_path": str(fixed_img_path),
        "registration_dir": str(registration_dir),
        "filename_moving_to_fixed": "moving_to_fixed.nii",
        "filename_fixed_to_moving": "fixed_to_moving.nii",
        "type_of_transform": "Rigid",
    }

@pytest.fixture
def ants_warp_image_test_args():
    """Arguments for ants_warp_image test."""
    # Define paths to required files
    moving_img_path = TEST_DATA_DIR / "TEMPLATES" / "Input" / "brain-segmentation.nii.gz"
    fixed_img_path = TEST_DATA_DIR / "PET" / "Input" / "pet_3d.nii"
    registration_dir = TEST_DATA_DIR / "tests" / "reg_test_results"
    output_path = registration_dir / "warped_brain_segmentation.nii"

    return {
        "moving_img_path": str(moving_img_path),
        "fixed_img_path": str(fixed_img_path),
        "output_path": str(output_path),
        "interpolator": "genericLabel",
    }

@pytest.fixture
def registration_module_namespace():
    """Fixture for providing a shared namespace between tests to share 
    transformation matrix between registration and warping tests."""
    return {"transformation_mat": None}

@pytest.fixture
def petpvc_create_4d_mask_test_args():
    """Arguments for petpvc_create_4d_mask test."""
    registration_dir = TEST_DATA_DIR / "tests" / "reg_test_results"
    template_path = registration_dir / "warped_brain_segmentation.nii"
    mask_4d_out_path = TEST_DATA_DIR / "tests" / "mask_4d_results" / "mask_4d.nii.gz"

    return {
        "template_path": str(template_path),
        "template_name": "FreeSurfer",
        "reference_name": "WholeCerebellum",
        "mask_4d_out": str(mask_4d_out_path)
    }

@pytest.fixture
def get_reference_region_mask_test_args():
    """Arguments for get_reference_region_mask test."""
    template_path = TEST_DATA_DIR / "TEMPLATES" / "Input" / "brain-segmentation.nii.gz"
    mask_out = TEST_DATA_DIR / "tests" / "reference_region_masks" / "reference_region.nii.gz"

    return {
        "template_path": str(template_path),
        "template_name": "FreeSurfer",
        "reference_name": "WholeCerebellum",
        "mask_out": str(mask_out)
    }

@pytest.fixture
def volume_dir(tmp_path):
    """Fixture to set up a directory with synthetic 3D volumes."""
    vol_dir = tmp_path / "3d_volumes"
    vol_dir.mkdir()
    
    # Create 3 synthetic 3D Nifti images
    for i in range(3):
        data = np.random.rand(64, 64, 64)  # Synthetic 3D data of shape 64x64x64
        affine = np.eye(4)  # Identity affine
        img = nib.Nifti1Image(data, affine)
        nib.save(img, vol_dir / f"volume_{i}.nii")

    return vol_dir

@pytest.fixture
def compute_4d_image_test_args(volume_dir, tmp_path):
    """Arguments for compute_4d_image test."""
    return {
        "volume_dir": volume_dir,
        "img_4d_out": str(tmp_path / "output_4d.nii")  # Temporary output file
    }

@pytest.fixture
def compute_mean_volume_test_args(volume_dir, tmp_path):
    """Arguments for compute_mean_volume test."""
    return {
        "volume_dir": volume_dir,
        "mean_3d_out": str(tmp_path / "output_mean_3d.nii")  # Temporary output file
    }

@pytest.fixture
def c3d_space_check_test_args():
    """Arguments for c3d_space_check test."""
    t1_mni_path = TEST_DATA_DIR / "T1" / "Input" / "t1_mni.nii"
    pet_3d_path = TEST_DATA_DIR / "PET" / "Input" / "pet_3d.nii"
    brain_seg_mni_path = TEST_DATA_DIR / "TEMPLATES" / "Input" / "brain-segmentation.nii.gz"

    return {
        "image1_path_case1": str(brain_seg_mni_path),
        "image2_path_case1": str(t1_mni_path),
        "image1_path_case2": str(pet_3d_path),
        "image2_path_case2": str(t1_mni_path),
    }

@pytest.fixture
def validate_settings_json_test_args():
    """Arguments for validate_dict_structure test"""
    # Paths to PET images
    pet_3d_image_path = TEST_DATA_DIR / "PET" / "Input" / "pet_3d.nii"
    pet_4d_image_path = TEST_DATA_DIR / "PET" / "Input" / "pet_4d.nii"
    # Sample of the VALID settings JSON template
    input_json_is_valid = {
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
            "InjectedRadioactivity": "200",
            "InjectedRadioactivityUnits": "MBq",
            "InjectionEnd": "40",
            "InjectionStart": 5,
            "Manufacturer": "GE",
            "ManufacturersModelName": "Discovery",
            "ModeOfAdministration": "bolus",
            "ReconFilterSize": "3.0",
            "ReconFilterType": "Gaussian",
            "ReconMethodName": "OSEM",
            "ReconMethodParameterLabels": "iterations",
            "ReconMethodParameterUnits": "none",
            "ReconMethodParameterValues": "120",
            "ScanStart": 0,
            "SpecificRadioactivity": "40",
            "SpecificRadioactivityUnits": "GBq/ug",
            "TimeZero": "10:00:00",
            "TracerName": "FDG",
            "TracerRadionuclide": "F18",
            "Units": "kBq/mL"
        }
    }
    # Example of the INVALID settings JSON template
    input_json_is_not_valid = {
        "pet_json": {
            "AcquisitionMode": "4D",
            "AttenuationCorrection": "Activity decay corrected",
            "BodyPart": "brain",
            "FrameDuration": "15, 15, 15, 15, 30, 30, 30, 30, 60, 60, 60, 60, 60,\
                180, 180, 180, 180, 300, 300, 300, 300, 300, 300, 300,\
                300, 300, 300, 300",
            "FrameTimesStart": [
                0, 15, 30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 420,
                480, 660, 840, 1020, 1200, 1500, 1800, 2100, 2400, 2700,
                3000, 3300, 3600, 3900
            ],
            "ImageDecayCorrected": "true",
            "ImageDecayCorrectionTime": "0",
            "InjectedMass": "5",
            "InjectedMassUnits": "ug",
            "InjectedRadioactivity": "200",
            "InjectedRadioactivityUnits": "MBq",
            "InjectionEnd": "40",
            "InjectionStart": 5,
            "Manufacturer": "GE",
            "ManufacturersModelName": "Discovery",
            "ModeOfAdministration": "bolus",
            "ReconFilterSize": "3.0",
            "ReconFilterType": "Gaussian",
            "ReconMethodName": "OSEM",
            "ReconMethodParameterLabels": "iterations",
            "ReconMethodParameterUnits": "none",
            "ReconMethodParameterValues": 120,
            "ScanStart": "0",
            "SpecificRadioactivity": "40",
            "SpecificRadioactivityUnits": "GBq/ug",
            "TimeZero": "10:00:00",
            "TracerName": "FDG",
            "TracerRadionuclide": "F18",
            "Units": "kBq/mL"
        }
    }
    # Version 2 of the invalid settings JSON file
    input_json_is_not_valid_v2 = copy.deepcopy(input_json_is_valid)
    input_json_is_not_valid_v2["pet_json"]["FrameTimesStart"][0] = 15
    return {
        "pet_3d_image_path": pet_3d_image_path,
        "pet_4d_image_path": pet_4d_image_path,
        "valid_json_input": input_json_is_valid,
        "invalid_json_input": input_json_is_not_valid,
        "invalid_json_input_v2": input_json_is_not_valid_v2        
    }