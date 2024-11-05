import pytest
import nibabel as nib
import numpy as np
from pathlib import Path

# Locate the PETScope-Test-Data directory relative to this file
TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "PETScope-Test-Data"

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