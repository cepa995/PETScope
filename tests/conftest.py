import pytest
import nibabel as nib
import numpy as np

@pytest.fixture
def ants_registration_test_args():
    """Arguments for ants_registration test."""
    return {
        "moving_img_path": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/data/t1_mni.nii",
        "fixed_img_path": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/data/pet_3d.nii",
        "registration_dir": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/reg_test_results",
        "filename_moving_to_fixed": "moving_to_fixed.nii",
        "filename_fixed_to_moving": "fixed_to_moving.nii",
        "type_of_transform": "Rigid",
    }

@pytest.fixture
def ants_warp_image_test_args():
    """Arguments for ants_warp_image test."""
    return {
        "moving_img_path": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/data/brain-segmentation.nii.gz",
        "fixed_img_path": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/data/pet_3d.nii",
        "output_path": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/reg_test_results/warped_brain_segmentation.nii.gz",
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
    return {
        "mask_3d_path": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/petscope-test-results-srtm/PET_TO_T1/brain_seg_pet_space.nii",
        "list_of_labels": [17, 53],
        "mask_4d_out": "/neuro/stefan/workspace/SVA2-PET/SRTM-PIPELINE/mask_4d_results/mask_4d.nii.gz"
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
