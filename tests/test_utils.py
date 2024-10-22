import os
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
