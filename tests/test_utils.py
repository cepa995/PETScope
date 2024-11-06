import os
import pytest
import numpy as np
import nibabel as nib
from petscope.utils import validate_settings_json
from petscope.exceptions import (
    SettingsJSONInvalidStructureException,
    PETImageNotFoundException,
    PET3DImageException,
    FrameNumberMismatchException,
    FrameStartTimeAndOrDurationException,
    PETDataUnitsException
)

def test_compute_4d_image(compute_4d_image_test_args) -> None:
    """Test creation of a 4D image from 3D volumes."""
    from petscope.utils import compute_4d_image

    # Run the function
    nifti_4d = compute_4d_image(**compute_4d_image_test_args)

    # Assertions
    assert nifti_4d is not None, "Output 4D image is None"
    assert len(nifti_4d.shape) == 4, "Expected 4D output for the concatenated image"
    assert os.path.exists(compute_4d_image_test_args["img_4d_out"]), "Output 4D file was not saved"
    assert nifti_4d.shape[-1] == len(os.listdir(compute_4d_image_test_args["volume_dir"])), \
        "Mismatch between 4D image time dimension and number of 3D input volumes"


def test_compute_mean_volume(compute_mean_volume_test_args) -> None:
    """Test creation of a mean 3D volume from 3D volumes."""
    from petscope.utils import compute_mean_volume

    # Run the function
    mean_3d_nii = compute_mean_volume(**compute_mean_volume_test_args)

    # Load a sample 3D volume to compare shapes
    img_3d_path = os.path.join(compute_mean_volume_test_args["volume_dir"], "volume_0.nii")
    img_3d_nii = nib.load(img_3d_path)

    # Assertions
    assert mean_3d_nii is not None, "Output mean 3D image is None"
    assert len(mean_3d_nii.shape) == 3, "Expected 3D output for the mean image"
    assert mean_3d_nii.shape == img_3d_nii.shape, "Mean 3D volume shape mismatch with input volumes"
    assert os.path.exists(compute_mean_volume_test_args["mean_3d_out"]), "Output mean 3D file was not saved"
    
    # Optional: Additional check for mean value (basic data integrity check)
    mean_data = mean_3d_nii.get_fdata()
    individual_data = [nib.load(os.path.join(compute_mean_volume_test_args["volume_dir"], f)).get_fdata()
                       for f in os.listdir(compute_mean_volume_test_args["volume_dir"])]
    calculated_mean = np.mean(individual_data, axis=0)
    assert np.allclose(mean_data, calculated_mean, atol=1e-5), "Mean 3D volume data mismatch"

def load_image(path):
    """Helper to check if the path exists and load the image."""
    assert os.path.exists(path), f"Image path does not exist: {path}"
    image = nib.load(path)
    assert image is not None, f"Failed to load image at: {path}"
    return image

@pytest.mark.parametrize("image1_key, image2_key, expected", [
    ("image1_path_case1", "image2_path_case1", True),
    ("image1_path_case2", "image2_path_case2", False),
])
def test_c3d_space_check(c3d_space_check_test_args, image1_key, image2_key, expected) -> None:
    """
    Test c3d_space_check function to verify if two images are in the same space.
    :param c3d_space_check_test_args: dictionary with test paths for images
    :param image1_key: Key for the first image path in the test args
    :param image2_key: Key for the second image path in the test args
    :param expected: Expected result (True if same space, False otherwise)
    """
    from petscope.utils import c3d_space_check

    image1_path = c3d_space_check_test_args[image1_key]
    image2_path = c3d_space_check_test_args[image2_key]

    # Load and verify images exist and are valid
    load_image(image1_path)
    load_image(image2_path)

    # Check if images are in the same space
    result = c3d_space_check(image1_path=image1_path, image2_path=image2_path)
    
    # Assert the result matches expected value
    assert result == expected, f"Expected {expected} but got {result} for {image1_key} and {image2_key}"

def test_get_reference_region_mask(get_reference_region_mask_test_args) -> None:
    """Tests get_reference_region mask function for producing a desired reference
    region """
    from petscope.utils import get_reference_region_mask

    # Parse arguments
    template_path = get_reference_region_mask_test_args["template_path"]
    template_name = get_reference_region_mask_test_args["template_name"]
    reference_name = get_reference_region_mask_test_args["reference_name"]
    mask_out = get_reference_region_mask_test_args["mask_out"]

    # Call/Test get_reference_region_mask function
    reference_region = get_reference_region_mask(
        template_path=template_path,
        template_name=template_name,
        reference_name=reference_name,
        mask_out=mask_out
    )
    reference_region_data = reference_region.get_fdata().astype(np.uint8)

    # Assertions
    assert reference_region
    assert len(reference_region_data.shape) == 3
    assert np.max(reference_region_data) == 1 and np.min(reference_region_data) == 0

def test_validate_settings_json_success(validate_settings_json_test_args):
    """Test that validate_settings_json runs successfully with valid input."""
    args = validate_settings_json_test_args
    assert validate_settings_json(args['pet_4d_image_path'], args['valid_json_input'])

def test_validate_settings_json_invalid_structure(validate_settings_json_test_args):
    """Test that validate_settings_json raises SettingsJSONInvalidStructureException for invalid input."""
    args = validate_settings_json_test_args
    with pytest.raises(SettingsJSONInvalidStructureException, match="Check data types and required"):
        validate_settings_json(args['pet_4d_image_path'], args['invalid_json_input'])

def test_validate_settings_json_pet_image_not_found(validate_settings_json_test_args):
    """Test that validate_settings_json raises PETImageNotFoundException for non-existent PET image path."""
    args = validate_settings_json_test_args
    with pytest.raises(PETImageNotFoundException, match="Path to the PET image does not exist"):
        validate_settings_json("invalid/path/to/pet_image.nii", args['valid_json_input'])

def test_validate_settings_json_pet_3d_image_exception(validate_settings_json_test_args):
    """Test that validate_settings_json raises PET3DImageException for 3D PET image."""
    args = validate_settings_json_test_args
    with pytest.raises(PET3DImageException, match="PET image should be 4D, got 3D instead"):
        validate_settings_json(args['pet_3d_image_path'], args['valid_json_input'])

def test_validate_settings_json_frame_number_mismatch(validate_settings_json_test_args, tmp_path):
    """Test that validate_settings_json raises FrameNumberMismatchException for frame mismatch."""
    args = validate_settings_json_test_args
    data = np.random.rand(64, 64, 64, 64)  
    affine = np.eye(4) 
    img = nib.Nifti1Image(data, affine)
    img_path = str(tmp_path / f"pet_4d_test.nii")
    nib.save(img, img_path)
    with pytest.raises(FrameNumberMismatchException, match="Found 64 time frames in PET 4D while in settings JSON found 27"):
        validate_settings_json(img_path, args['valid_json_input'])

def test_validate_settings_json_frame_disagreement(validate_settings_json_test_args):
    """Test that validate_settings_json raises FrameStartTimeAndOrDurationException exception."""
    args = validate_settings_json_test_args
    with pytest.raises(FrameStartTimeAndOrDurationException, match="There is a disagreement between "
                       + "frame start time and frame duration lists in settings_template.json"):
        validate_settings_json(args["pet_4d_image_path"], args["invalid_json_input_v2"])

def test_validate_settings_json_kbqml_units(validate_settings_json_test_args):
    """Test that validate_settings_json runs successfully with valid input."""
    args = validate_settings_json_test_args
    invalid_json_input = args["valid_json_input"]
    invalid_json_input["pet_json"]["Units"] = "mm3"
    with pytest.raises(PETDataUnitsException, match="Expected kBq/mL units, got mm3 instead"):
        validate_settings_json(args["pet_4d_image_path"], invalid_json_input)