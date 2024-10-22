import os
import ants
import pytest

@pytest.mark.dependency()
def test_ants_registration(ants_registration_test_args) -> None:
    """Test ANTs registration and save transformation matrix path."""
    from petscope.registration import ants_registration

    transformation = ants_registration(**ants_registration_test_args)

    warped_moving_img_path = os.path.join(
        ants_registration_test_args["registration_dir"], 
        ants_registration_test_args["filename_moving_to_fixed"]
    )
    warped_fixed_img_path = os.path.join(
        ants_registration_test_args["registration_dir"], 
        ants_registration_test_args["filename_fixed_to_moving"]
    )

    # Assert files were created and contain data
    assert os.path.exists(warped_moving_img_path) and os.path.getsize(warped_moving_img_path) > 0
    assert os.path.exists(warped_fixed_img_path) and os.path.getsize(warped_fixed_img_path) > 0
    assert os.path.exists(transformation) and os.path.getsize(transformation) > 0


@pytest.mark.dependency(depends=['test_ants_registration'])
def test_ants_warp_image(ants_warp_image_test_args, ants_registration_test_args) -> None:
    """Test ANTs Warp Image after registration, using both forward and inverse transforms."""
    from petscope.registration import ants_warp_image

    transform_path = [os.path.join(ants_registration_test_args["registration_dir"], f)
                        for f in os.listdir(ants_registration_test_args["registration_dir"]) if f.endswith('.mat')][0]
    assert transform_path, "Transformation matrix path is missing from prior test."

    # Apply forward transform
    ants_warp_image(
        transform_path=transform_path,
        **ants_warp_image_test_args
    )
    assert os.path.exists(ants_warp_image_test_args["output_path"]) and os.path.getsize(ants_warp_image_test_args["output_path"]) > 0

    # Apply inverse transform
    ants_warp_image(
        transform_path=transform_path,
        is_inverse=True,
        **ants_warp_image_test_args
    )
    assert os.path.exists(ants_warp_image_test_args["output_path"]) and os.path.getsize(ants_warp_image_test_args["output_path"]) > 0
