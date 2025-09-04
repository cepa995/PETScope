import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich import print
from petscope.kinetic_modeling.srtm2 import srtm2, calculate_weights, estimate_k2prime, estimate_k2prime_voxelwise
from petscope.kinetic_modeling.utils import prepare_pet_data, create_masks, prepare_rois, \
      save_results, diagnostic_plots, prepare_roi_masks
from petscope.dynamicpet_wrapper.srtm import compute_target_region_stats


def compute_DVR_image(pet_file, output_dir, frame_durations, template_path, template_name, reference_region, 
                        roi_regions=None, multstart_iter=100, k2prime_method='voxel_based', verbose=True):
    """
    Process a complete PET study to generate DVR images using the two-pass SRTM2 approach.
    This is the main controller function that coordinates the entire process.
    
    Args:
        pet_file (str): Path to 4D PET NIFTI file
        output_dir (str): Directory for output files
        frame_durations (array_like): Duration of each frame in seconds
        template_path (str): Path to template/atlas for region definition
        template_name (str): Name of the template/atlas
        reference_region (str): Name of the reference region (e.g., 'cerebellum')
        roi_regions (list, optional): List of ROI names for k2prime estimation
        multstart_iter (int, optional): Number of multi-start iterations for ROI fitting
        k2prime_method (str, optional): Method for k2prime estimation. Options:
            - 'tac_based': Use ROI-averaged TACs (faster, current method)
            - 'voxel_based': Use individual voxels within ROIs (more rigorous)
        verbose (bool, optional): Whether to print detailed progress information
    
    Returns:
        dict: Dictionary containing paths to output files and parameter estimates
    """
    # Validate k2prime_method parameter
    valid_methods = ['tac_based', 'voxel_based']
    if k2prime_method not in valid_methods:
        raise ValueError(f"k2prime_method must be one of {valid_methods}, got '{k2prime_method}'")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"Using k2prime estimation method: {k2prime_method}")
    
    # 1. Prepare PET data 
    pet_info = prepare_pet_data(pet_file, frame_durations, verbose)
    
    # 2. Create masks 
    masks_info = create_masks(
        pet_info['pet_data'], 
        template_path, 
        template_name, 
        reference_region, 
        output_dir, 
        verbose
    )
    
    # 3. Prepare ROIs for k2prime estimation
    if k2prime_method == 'tac_based':
        # Prepare averaged TACs
        roi_info = prepare_rois(
            pet_info['pet_data'], 
            masks_info['brain_mask'], 
            roi_regions, 
            template_path, 
            template_name, 
            output_dir, 
            pet_info['affine'], 
            pet_info['header'], 
            pet_info['t_tac'],
            verbose
        )
        
        # 4. First pass - estimate k2prime from ROI TACs
        k2prime_info = estimate_k2prime(
            t_tac=pet_info['t_tac'], 
            ref_tac=masks_info['ref_tac'], 
            roi_tacs=roi_info['roi_tacs'], 
            multstart_iter=multstart_iter, 
            frame_durations=frame_durations,
            verbose=verbose,
        )
        
    elif k2prime_method == 'voxel_based':
        # New method: prepare ROI masks for voxel-wise analysis
        roi_info = prepare_roi_masks(
            masks_info['brain_mask'], 
            roi_regions, 
            template_path, 
            template_name, 
            output_dir, 
            verbose
        )
        
        # 4. First pass - estimate k2prime from individual voxels
        k2prime_info = estimate_k2prime_voxelwise(
            pet_data=pet_info['pet_data'],
            t_tac=pet_info['t_tac'], 
            ref_tac=masks_info['ref_tac'], 
            roi_masks=roi_info['roi_masks'], 
            multstart_iter=multstart_iter, 
            frame_durations=frame_durations,
            verbose=verbose,
        )
    
    # 5. Second pass - create parametric images with fixed k2prime
    parametric_images = create_parametric_images(
        pet_data=pet_info['pet_data'], 
        t_tac=pet_info['t_tac'], 
        ref_tac=masks_info['ref_tac'], 
        brain_mask=masks_info['brain_mask'], 
        global_k2prime=k2prime_info['global_k2prime'], 
        frame_durations=frame_durations,
        verbose=verbose
    )
    
    # 6. Save results 
    output_paths = save_results(
        parametric_images, 
        pet_info['affine'], 
        pet_info['header'], 
        output_dir, 
        roi_info['roi_tacs'] if k2prime_method == 'tac_based' else None, 
        k2prime_info['roi_results'], 
        k2prime_info['k2prime_values'], 
        k2prime_info['global_k2prime'], 
        verbose
    )
    
    # 7. Create comprehensive diagnostic plots
    diag_path = diagnostic_plots(
        parametric_images, 
        k2prime_info['global_k2prime'], 
        k2prime_info['roi_results'],
        output_dir, 
        verbose
    )

    # 8. Compute statistics 
    dvr_path = os.path.join(output_dir, 'srtm2_dvr.nii.gz')
    if roi_regions:
        for target_region in roi_regions:
            compute_target_region_stats(
                dvr_path=dvr_path,
                template_path=template_path,
                template_name=template_name,
                target_region=target_region,
                output_dir=output_dir
            )

    if verbose:
        print(f"\n:white_check_mark: SRTM2 DVR processing complete. Results saved to {output_dir}")
        print(f":bar_chart: Diagnostic plots: {diag_path}")
    
    return {
        'dvr_path': output_paths['dvr_path'],
        'bp_path': output_paths['bp_path'],
        'R1_path': output_paths['R1_path'],
        'k2a_path': output_paths['k2a_path'],
        'k2prime': k2prime_info['global_k2prime'],
        'roi_results': k2prime_info['roi_results'],
        'diagnostic_path': diag_path
    }

def create_parametric_images(pet_data, t_tac, ref_tac, brain_mask, global_k2prime, frame_durations, verbose=True):
    """
    Second pass: Create parametric images with fixed k2prime.
    
    Args:
        pet_data (ndarray): 4D PET data array
        t_tac (ndarray): Time points array
        ref_tac (ndarray): Reference region TAC
        brain_mask (ndarray): Brain mask array
        global_k2prime (float): Global k2prime value from first pass
        verbose (bool): Flag to enable detailed output
        
    Returns:
        dict: Dictionary containing parametric images
    """
    if verbose:
        print("\nSecond pass: Generating voxel-wise parametric images...")
        print(f"Using fixed k2prime: {global_k2prime:.4f}")
    
    # Initialize output parameter volumes
    shape = pet_data.shape[:3]
    R1_img = np.zeros(shape)
    bp_img = np.zeros(shape)
    dvr_img = np.zeros(shape)
    k2a_img = np.zeros(shape)
    
    R1_bounds = (0.5, 2.0)  # Tighter range than (0.0, 5.0)
    bp_bounds = (-0.5, 8.0)  # Allow slightly negative but not extreme values
    
    # Count masked voxels for progress tracking
    total_voxels = np.sum(brain_mask)
    voxel_counter = 0
    fail_counter = 0
    bound_hit_counter = 0
    
    if verbose:
        print(f"Processing {total_voxels} voxels with bounds:")
        print(f"  R1: [{R1_bounds[0]}, {R1_bounds[1]}]")
        print(f"  BP: [{bp_bounds[0]}, {bp_bounds[1]}]")
    
    # Loop through voxels within the brain mask
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if brain_mask[x, y, z]:
                    voxel_counter += 1
                    
                    # Show progress periodically
                    if verbose and voxel_counter % 5000 == 0:
                        progress = 100 * voxel_counter / total_voxels
                        fail_rate = 100 * fail_counter / voxel_counter if voxel_counter > 0 else 0
                        bound_rate = 100 * bound_hit_counter / voxel_counter if voxel_counter > 0 else 0
                        print(f"  Progress: {voxel_counter}/{total_voxels} ({progress:.1f}%)")
                        print(f"    Failed fits: {fail_counter} ({fail_rate:.1f}%)")
                        print(f"    Boundary hits: {bound_hit_counter} ({bound_rate:.1f}%)")
                    
                    # Extract voxel TAC
                    voxel_tac = pet_data[x, y, z, :]
                    
                    # Pre-screen voxels with very low activity
                    mean_activity = np.mean(voxel_tac)
                    ref_activity = np.mean(ref_tac)
                    if mean_activity < 0.1 * ref_activity:  # Very low activity
                        R1_img[x, y, z] = 0
                        bp_img[x, y, z] = 0
                        k2a_img[x, y, z] = 0
                        dvr_img[x, y, z] = 1  # No binding
                        fail_counter += 1
                        continue
                    
                    try:
                        weights = calculate_weights(frame_durations, voxel_tac) if frame_durations is not None else np.ones_like(t_tac)
                        
                        # Use tighter bounds and multiple starts for difficult voxels
                        result = srtm2(
                            t_tac=t_tac,
                            reftac=ref_tac,
                            roitac=voxel_tac,
                            k2prime=global_k2prime,  # Fixed k2prime from first pass (result of estimate_k2primne function)
                            multstart_iter=3,  # Use multiple starts for better convergence
                            printvals=False,
                            weights=weights,
                            R1_start=1.0, 
                            R1_lower=R1_bounds[0], 
                            R1_upper=R1_bounds[1],
                            bp_start=0.5, 
                            bp_lower=bp_bounds[0], 
                            bp_upper=bp_bounds[1]
                        )
                        
                        # Extract fitted parameters
                        R1_val = result['par']['R1'].values[0]
                        bp_val = result['par']['bp'].values[0]
                        k2a_val = result['par']['k2a'].values[0]
                        dvr_val = bp_val + 1.0
                        
                        # Check if parameters hit bounds (indicates potential problems)
                        tolerance = 1e-6
                        if (abs(R1_val - R1_bounds[0]) < tolerance or 
                            abs(R1_val - R1_bounds[1]) < tolerance or
                            abs(bp_val - bp_bounds[0]) < tolerance or 
                            abs(bp_val - bp_bounds[1]) < tolerance):
                            bound_hit_counter += 1
                        
                        # Store parameter values
                        R1_img[x, y, z] = R1_val
                        bp_img[x, y, z] = bp_val
                        k2a_img[x, y, z] = k2a_val
                        dvr_img[x, y, z] = dvr_val
                        
                    except Exception as e:
                        # Handle fitting failures - use default values
                        fail_counter += 1
                        if verbose and fail_counter % 1000 == 0:
                            print(f"    Fitting error at ({x},{y},{z}): {str(e)[:50]}...")
                        
                        R1_img[x, y, z] = 0
                        bp_img[x, y, z] = 0
                        k2a_img[x, y, z] = 0
                        dvr_img[x, y, z] = 1  # No binding
    
    if verbose:
        print(f"\nVoxel-wise fitting summary:")
        print(f"  Total voxels processed: {voxel_counter}")
        fail_percentage = 100 * fail_counter / voxel_counter if voxel_counter > 0 else 0
        bound_percentage = 100 * bound_hit_counter / voxel_counter if voxel_counter > 0 else 0
        print(f"  Failed fits: {fail_counter} ({fail_percentage:.1f}%)")
        print(f"  Boundary hits: {bound_hit_counter} ({bound_percentage:.1f}%)")
        
        valid_mask = brain_mask & (R1_img > 0)
        if np.sum(valid_mask) > 0:
            print(f"  Parameter ranges in valid voxels:")
            print(f"    R1: [{np.min(R1_img[valid_mask]):.3f}, {np.max(R1_img[valid_mask]):.3f}]")
            print(f"    BP: [{np.min(bp_img[valid_mask]):.3f}, {np.max(bp_img[valid_mask]):.3f}]")
            print(f"    DVR: [{np.min(dvr_img[valid_mask]):.3f}, {np.max(dvr_img[valid_mask]):.3f}]")
    
    return {
        'R1_img': R1_img,
        'bp_img': bp_img,
        'k2a_img': k2a_img,
        'dvr_img': dvr_img
    }