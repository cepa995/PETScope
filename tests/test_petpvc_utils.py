import os
import pytest

def test_petpvc_create_4d_mask(petpvc_create_4d_mask_test_args):
    """ Test 4D Mask creation, required for PET Partial Volume Correction """
    from petscope.petpvc_wrapper.utils import petpvc_create_4d_mask
    mask_4d_nii = petpvc_create_4d_mask(**petpvc_create_4d_mask_test_args)
    # Assert results
    assert mask_4d_nii and os.path.exists(petpvc_create_4d_mask_test_args["mask_4d_out"]) and os.path.getsize(petpvc_create_4d_mask_test_args["mask_4d_out"]) > 0 