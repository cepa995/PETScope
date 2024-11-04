import os
import pytest
import numpy as np
import nibabel as nib

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