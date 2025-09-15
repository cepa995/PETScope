import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print
from scipy.optimize import least_squares
from scipy.interpolate import interp1d, UnivariateSpline
from petscope.kinetic_modeling.utils import fix_multstartpars, kinfit_convolve, calculate_weights

# Specific region bounds
region_bounds = {}

# Default bounds for any other regions not specifically defined
default_bounds = {
    "R1": {"start": 1.0, "lower": 0.2, "upper": 3.0},      # Slightly wider range
    "k2prime": {"start": 0.15, "lower": 0.02, "upper": 0.3},
    "bp": {"start": 2.0, "lower": -0.5, "upper": 8.0}       # Higher starting point
}

def process_srtm2_output(result, modeldata, upper, lower, k2prime=None, verbose=False):
    """
    Processes the output of SRTM2 model fitting to check parameter limits, 
    compute fitted TACs, and return structured output.

    Arguments:
        result (dict): The output from fit_srtm2_model containing fitted parameters.
        modeldata (dict): Dictionary containing 't_tac', 'roitac', 'reftac'.
        upper (dict): Upper bounds of parameters.
        lower (dict): Lower bounds of parameters.
        k2prime (float, optional): If not None, the k2prime parameter is fixed.
        verbose (boolean): If set to True, it will print out warning if the fitted parameters
        are hitting upper and/or lower limit bounds.

    Returns
        dict :A dictionary containing fitted parameters, standard errors, fitted TACs, 
            and metadata about the model fit.
    """
    fitted_params = np.round(np.array(list(result.values())), 3)
    if k2prime is None:
        upper_bounds = np.round(np.array([upper[key] for key in result.keys()]), 3)
        lower_bounds = np.round(np.array([lower[key] for key in result.keys()]), 3)
    else:
        upper_bounds = np.round(np.array([upper[key] if key != "k2prime" else k2prime for key in result.keys()]), 3)
        lower_bounds = np.round(np.array([lower[key] if key != "k2prime" else k2prime for key in result.keys()]), 3)

    # Check if parameters hit upper or lower limits
    limcheck_u = np.equal(fitted_params, upper_bounds)
    limcheck_l = np.equal(fitted_params, lower_bounds)
    limcheck = limcheck_u | limcheck_l

    if np.any(limcheck) and verbose:
        print("\nWarning: Fitted parameters are hitting upper or lower limit bounds. "
              "Consider modifying the upper and lower limit boundaries, "
              "or using a multi-start approach when fitting the model.\n")

    # Create DataFrame for TACs
    tacs = pd.DataFrame({
        "Time": modeldata["t_tac"],
        "Reference": modeldata["reftac"],
        "Target": modeldata["roitac"],
        "Target_fitted": srtm2_model(modeldata["t_tac"], modeldata["reftac"], **result)
    })

    # Convert parameters to DataFrame
    par = pd.DataFrame([result])

    # Placeholder function for standard error estimation
    def get_se(param_name):
        return 0.0  # TODO: Replace with real SE calculation if available

    # Compute standard errors
    par_se = par.copy()
    par_se.iloc[0] = [get_se(param) for param in result.keys()]
    par_se.columns = [f"{col}.se" for col in par_se.columns]

    # Add fixed k2prime if provided
    if k2prime is not None:
        par["k2prime"] = k2prime
        par_se["k2prime.se"] = 0

    # Compute derived parameter k2a
    par["k2a"] = (par["R1"] * par["k2prime"]) / (par["bp"] + 1)
    par_se["k2a.se"] = get_se("k2a")

    # Add DVR calculation
    par["dvr"] = par["bp"] + 1
    par_se["dvr.se"] = par_se["bp.se"]  # Same uncertainty as BP
    
    # Construct final output
    output = {
        "par": par,
        "par_se": par_se,
        "fit": result,
        "weights": modeldata["weights"],
        "tacs": tacs,
        "model": "srtm2"
    }
    
    return output

def srtm2_model(t_tac, reftac, R1, k2prime, bp, interpolation_method='cubic', n_interp_points=512):
    """
    Simplified Reference Tissue Model 2 (SRTM2) with improved interpolation
    
    Arguments:
        t_tac (array_like): 1D array of time points (in minutes). (We use the time at mid‐frame.)
        reftac (array_like): 1D array of radioactivity concentrations in the reference tissue.
        R1 (float): Parameter R1.
        k2prime (float): Parameter k2prime.
        bp (float): Parameter bp.
        interpolation_method (str): Interpolation method ('linear', 'cubic', 'exponential', 'spline')
        n_interp_points (int): Number of interpolation points (default 512)
        
    Returns:
        outtac (ndarray): 1D array of the predicted target tissue time-activity curve.
    """
    # Create interpolated time grid
    interptime = np.linspace(np.min(t_tac), np.max(t_tac), n_interp_points)  
    step = interptime[1] - interptime[0]

    if step <= 0 or step < 1e-10:
        raise ValueError("Step size is too small!")

    # Choose interpolation method
    if interpolation_method == 'linear':
        iref = np.interp(interptime, t_tac, reftac)
        
    elif interpolation_method == 'cubic':
        # Cubic spline interpolation
        if len(t_tac) < 4:
            # Fall back to linear if insufficient points for cubic
            iref = np.interp(interptime, t_tac, reftac)
        else:
            interp_func = interp1d(t_tac, reftac, kind='cubic', 
                                 bounds_error=False, fill_value='extrapolate')
            iref = interp_func(interptime)
            
    elif interpolation_method == 'exponential':
        # Exponential interpolation - fit exponential decay between points
        iref = exponential_interpolation(t_tac, reftac, interptime)
        
    elif interpolation_method == 'spline':
        # Smoothing spline - automatically determines smoothness
        if len(t_tac) < 4:
            iref = np.interp(interptime, t_tac, reftac)
        else:
            spline = UnivariateSpline(t_tac, reftac, s=None)  # s=None for interpolating spline
            iref = spline(interptime)
            
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    
    # Ensure no negative values (physiologically unrealistic)
    iref = np.maximum(iref, 0)
    
    # SRTM2 calculations
    k2a = (R1 * k2prime) / (bp + 1)
    
    a = R1 * (k2prime - k2a) * iref
    b = np.exp(-k2a * interptime)

    ND = R1 * iref
    BOUND = kinfit_convolve(a, b, step)
    i_outtac = ND + BOUND

    # Interpolate back to original time points
    if interpolation_method in ['cubic', 'spline'] and len(t_tac) >= 4:
        # Use same method for back-interpolation
        if interpolation_method == 'cubic':
            back_interp = interp1d(interptime, i_outtac, kind='cubic',
                                 bounds_error=False, fill_value='extrapolate')
        else:  # spline
            back_interp = UnivariateSpline(interptime, i_outtac, s=None)
        outtac = back_interp(t_tac)
    else:
        # Linear back-interpolation for other methods or insufficient points
        outtac = np.interp(t_tac, interptime, i_outtac)
        
    return outtac


def exponential_interpolation(t_orig, y_orig, t_new):
    """
    Exponential interpolation for PET TAC data
    
    Fits piecewise exponential functions between consecutive points,
    which is more physiologically appropriate for tracer kinetics.
    """
    y_new = np.zeros_like(t_new)
    
    for i in range(len(t_new)):
        t_target = t_new[i]
        
        # Find the interval containing t_target
        if t_target <= t_orig[0]:
            y_new[i] = y_orig[0]
        elif t_target >= t_orig[-1]:
            y_new[i] = y_orig[-1]
        else:
            # Find bracketing points
            idx = np.searchsorted(t_orig, t_target) - 1
            t1, t2 = t_orig[idx], t_orig[idx + 1]
            y1, y2 = y_orig[idx], y_orig[idx + 1]
            
            # Handle edge cases
            if y1 <= 0 or y2 <= 0:
                # Fall back to linear interpolation for non-positive values
                alpha = (t_target - t1) / (t2 - t1)
                y_new[i] = y1 + alpha * (y2 - y1)
            else:
                # Exponential interpolation: y = y1 * exp(k * (t - t1))
                # where k = ln(y2/y1) / (t2 - t1)
                if abs(y2 - y1) < 1e-10:  # Nearly constant
                    y_new[i] = y1
                else:
                    k = np.log(y2 / y1) / (t2 - t1)
                    y_new[i] = y1 * np.exp(k * (t_target - t1))
    
    return y_new

def adaptive_interpolation_points(t_tac, base_points=256, max_points=1024):
    """
    Adaptively determine number of interpolation points based on scan characteristics
    
    Arguments:
        t_tac: Original time points
        base_points: Minimum number of points
        max_points: Maximum number of points
        
    Returns:
        n_points: Optimal number of interpolation points
    """
    scan_duration = t_tac[-1] - t_tac[0]
    n_orig_points = len(t_tac)
    
    # More points for longer scans or sparser original sampling
    density_factor = scan_duration / n_orig_points
    n_points = int(base_points * (1 + density_factor / 10))
    
    return min(max_points, max(base_points, n_points))


def srtm2_model_adaptive(t_tac, reftac, R1, k2prime, bp, interpolation_method='cubic'):
    """
    SRTM2 model with adaptive interpolation point selection
    """
    n_interp = adaptive_interpolation_points(t_tac)
    return srtm2_model(t_tac, reftac, R1, k2prime, bp, 
                      interpolation_method=interpolation_method,
                      n_interp_points=n_interp)


def fit_srtm2_model(modeldata, start, lower, upper, weights,
                    multstart_iter, multstart_lower, multstart_upper,
                    printvals=False, k2prime_fixed=None):
    """
    Fit the Simplified Reference Tissue Model 2 (SRTM2) to data.

    Arguments:
        modeldata (dict or DataFrame): Dictionary or DataFrame containing:
            - 't_tac': 1D array of time points (in minutes).
            - 'roitac': 1D array of radioactivity concentrations in the target tissue.
            - 'reftac': 1D array of radioactivity concentrations in the reference tissue.
        start (dict or list): Starting values for the parameters. 
            If `k2prime_fixed` is None, expected keys are 'R1', 'k2prime', and 'bp'; otherwise, 'R1' and 'bp'.
        lower (dict or list): Lower bounds for parameter values (same structure as `start`).
        upper (dict or list): Upper bounds for parameter values (same structure as `start`).
        weights (array_like): 1D array of weights assigned to each data point.
        multstart_iter (int): Number of iterations for multi-start optimization. 
            If 1, a single fit is performed; otherwise, multiple starting values are used.
        multstart_lower (array_like): Lower bounds for the multi-start random parameter initialization.
        multstart_upper (array_like): Upper bounds for the multi-start random parameter initialization.
        printvals (bool, optional): If True, prints iteration messages for debugging.
        k2prime_fixed (float or None, optional): If provided, `k2prime` is fixed at this value.

    Returns:
        result (dict): Dictionary containing the optimized parameters:
            - 'R1' (float): Estimated R1 parameter.
            - 'k2prime' (float): Estimated or fixed k2prime parameter.
            - 'bp' (float): Estimated bp parameter.
    """
    # Make sure our data are numpy arrays
    t_tac = np.asarray(modeldata['t_tac'])
    roitac = np.asarray(modeldata['roitac'])
    reftac = np.asarray(modeldata['reftac'])
    
    # Validate input data
    if len(t_tac) != len(roitac) or len(t_tac) != len(reftac):
        raise ValueError("Time and TAC arrays must have the same length")
    
    if k2prime_fixed is not None:
        # Two-parameter fit (R1, bp)
        param_names = ['R1', 'bp']
        
        # Set up parameters consistently
        if isinstance(start, dict):
            p0 = [start['R1'], start['bp']]
            lower_bounds = [lower['R1'], lower['bp']]
            upper_bounds = [upper['R1'], upper['bp']]
        else:
            p0 = start[:2] if len(start) >= 2 else start
            lower_bounds = lower[:2] if len(lower) >= 2 else lower
            upper_bounds = upper[:2] if len(upper) >= 2 else upper
    else:
        # Three-parameter fit (R1, k2prime, bp)
        param_names = ['R1', 'k2prime', 'bp']
        
        if isinstance(start, dict):
            p0 = [start['R1'], start['k2prime'], start['bp']]
            lower_bounds = [lower['R1'], lower['k2prime'], lower['bp']]
            upper_bounds = [upper['R1'], upper['k2prime'], upper['bp']]
        else:
            p0 = start
            lower_bounds = lower
            upper_bounds = upper

    # Define residual function
    def residuals(params, t_tac, roitac, reftac, weights):
        try:
            if k2prime_fixed is None:
                R1, k2prime, bp = params
            else:
                R1, bp = params
                k2prime = k2prime_fixed
            
            model_pred = srtm2_model_adaptive(t_tac, reftac, R1, k2prime, bp, 'linear')
            residuals_arr = (model_pred - roitac) * weights
            
            # Check for invalid results
            if np.isnan(residuals_arr).any() or np.isinf(residuals_arr).any():
                return np.full_like(residuals_arr, 1e6)  # Large penalty
                
            return residuals_arr
            
        except Exception:
            # Return large residuals if model evaluation fails
            return np.full_like(roitac, 1e6) * weights

    # Choose between single fit or multi-start approach
    if multstart_iter == 1:
        # Single optimization
        try:
            result = least_squares(residuals, p0, 
                                 args=(t_tac, roitac, reftac, weights), 
                                 bounds=(lower_bounds, upper_bounds))
            if result.success:
                optimized_parameters = result.x
            else:
                raise RuntimeError(f"Optimization failed: {result.message}")
        except Exception as e:
            raise RuntimeError(f"Single optimization failed: {e}")
    else:
        # Multi-start optimization
        best_cost = np.inf
        optimized_parameters = None
        best_result = None
        
        for i in range(multstart_iter):
            try:
                # Generate random starting point with consistent parameter order
                if isinstance(multstart_lower, dict) and isinstance(multstart_upper, dict):
                    p0_trial = [np.random.uniform(multstart_lower[name], multstart_upper[name]) 
                               for name in param_names]
                else:
                    # Fallback to list-based generation
                    n_params = len(param_names)
                    p0_trial = [np.random.uniform(multstart_lower[j], multstart_upper[j]) 
                               for j in range(n_params)]
                
                result = least_squares(residuals, p0_trial, 
                                     args=(t_tac, roitac, reftac, weights), 
                                     bounds=(lower_bounds, upper_bounds))
                
                if result.success:
                    cost = np.sum(result.fun ** 2)
                    if printvals:
                        print(f"Iteration {i}: cost = {cost:.6f}, success = {result.success}")
                    
                    if cost < best_cost:
                        best_cost = cost
                        optimized_parameters = result.x
                        best_result = result
                else:
                    if printvals:
                        print(f"Iteration {i}: optimization failed - {result.message}")
                        
            except Exception as e:
                if printvals:
                    print(f"Iteration {i}: fit failed with {type(e).__name__}: {e}")
                continue
        
        # Check if any optimization succeeded
        if optimized_parameters is None:
            raise RuntimeError("All multi-start optimization attempts failed")
        
        if printvals:
            print(f"Best result: cost = {best_cost:.6f}, nfev = {best_result.nfev}")

    # Return results with consistent parameter names
    if k2prime_fixed is not None:
        return {
            'R1': optimized_parameters[0], 
            'k2prime': k2prime_fixed, 
            'bp': optimized_parameters[1]
        }
    else:
        return {
            'R1': optimized_parameters[0], 
            'k2prime': optimized_parameters[1], 
            'bp': optimized_parameters[2]
        }

def srtm2(t_tac, reftac, roitac, k2prime=None, weights=None, frame_start_end=None,
          R1_start=1, R1_lower=0, R1_upper=10,
          k2prime_start=0.1, k2prime_lower=0.001, k2prime_upper=1,
          bp_start=1.5, bp_lower=0, bp_upper=15,
          multstart_iter=1, multstart_lower=None, multstart_upper=None,
          printvals=False, verbose=False):
    """
    Fit the Simplified Reference Tissue Model 2 (SRTM2) to data.

    Arguments:
        t_tac (array_like): 1D array of time points (in minutes). (We use the time at mid‐frame.)
        reftac (array_like): 1D array of radioactivity concentrations in the reference tissue.
        roitac (array_like): 1D array of radioactivity concentrations in the target tissue.
        k2prime (float, optional): If None, the model will estimate k2prime. 
            If specified, the model will be fitted with this fixed value.
        weights (array_like, optional): 1D array of weights assigned to each data point.
            If not specified, uniform weights are used.
        frame_start_end (tuple, optional): Tuple (start_frame, end_frame) to specify the 
            subset of frames to use for modeling (e.g., (1, 20)).

        R1_start (float, optional): Initial guess for R1 parameter. Default is 1.
        R1_lower (float, optional): Lower bound for R1. Default is 0.
        R1_upper (float, optional): Upper bound for R1. Default is 10.

        k2prime_start (float, optional): Initial guess for k2prime parameter. Default is 0.1.
        k2prime_lower (float, optional): Lower bound for k2prime. Default is 0.001.
        k2prime_upper (float, optional): Upper bound for k2prime. Default is 1.

        bp_start (float, optional): Initial guess for binding potential (bp). Default is 1.5.
        bp_lower (float, optional): Lower bound for bp. Default is 0.
        bp_upper (float, optional): Upper bound for bp. Default is 15.

        multstart_iter (int, optional): Number of iterations for multi-start optimization.
            If 1, a single fit is performed; otherwise, multiple starting values are used.
        multstart_lower (dict, optional): Lower bounds for multi-start parameter initialization.
        multstart_upper (dict, optional): Upper bounds for multi-start parameter initialization.

        printvals (bool, optional): If True, prints iteration messages for debugging.

    Returns:
        output (dict): Dictionary containing:
            - 'par' (DataFrame): Fitted parameters (R1, k2prime, bp).
            - 'par_se' (DataFrame): Standard errors of fitted parameters.
            - 'fit' (dict): The model fit result.
            - 'weights' (array_like): The model weights.
            - 'tacs' (DataFrame): Time-activity curves (measured and fitted values).

    References:
        Wu Y, Carson RE. Noise reduction in the simplified reference tissue model for neuroreceptor functional imaging.
        J Cereb Blood Flow Metab. 2002;22:1440-1452.
    """
    # Step 0. Parse function arguments and check if k2prime has been specified
    if k2prime is None:
        if verbose:
            print("Note: Without specifying a k2prime value for SRTM2, it is effectively "
                "equivalent to the conventional SRTM model. This can be useful for "
                "selecting an appropriate k2prime value, but without re-fitting the "
                "model with a specified k2prime value, the model is not really SRTM2.")

        start = {'R1': R1_start, 'k2prime': k2prime_start, 'bp': bp_start}
        lower = {'R1': R1_lower, 'k2prime': k2prime_lower, 'bp': bp_lower}
        upper = {'R1': R1_upper, 'k2prime': k2prime_upper, 'bp': bp_upper}
    else:
        if isinstance(k2prime, (list, tuple)) and len(k2prime) > 1:
            raise ValueError("k2prime must be specified by a single value.")

        start = {'R1': R1_start, 'bp': bp_start}
        lower = {'R1': R1_lower, 'bp': bp_lower}
        upper = {'R1': R1_upper, 'bp': bp_upper}

    # Step 1: Check & Fix multstart parameters - ensure the correnct length for multstart bounds
    multstart_pars = fix_multstartpars(
        start=start,
        lower=lower,
        upper=upper,
        multstart_iter=multstart_iter,
        multstart_lower=multstart_lower,
        multstart_upper=multstart_upper
    )
    multstart_lower = multstart_pars['multstart_lower']
    multstart_upper = multstart_pars['multstart_upper']

    # Extract data from modeldata
    t_tac = np.array(t_tac)
    reftac = np.array(reftac)
    roitac = np.array(roitac)

    weights = np.array(weights) if weights is not None else np.ones_like(t_tac)

    # Bundle data into a dict:
    modeldata = {'t_tac': t_tac, 'roitac': roitac, 'reftac': reftac, 'weights': weights}
    # Fit the model (here with all 3 parameters free)
    result = fit_srtm2_model(modeldata, start, lower, upper,
                            weights=weights,
                            multstart_iter=multstart_iter,
                            multstart_lower=multstart_lower,
                            multstart_upper=multstart_upper,
                            printvals=False,
                            k2prime_fixed=k2prime)
    
    # Process SRTM output
    output = process_srtm2_output(result, modeldata, upper, lower, k2prime)
    return output

def estimate_k2prime_voxelwise(pet_data, t_tac, ref_tac, roi_masks,
                               multstart_iter=100, frame_durations=None, verbose=True,
                               k2_prime_threshold=0.03, k2_prime_upper_limit=0.4,
                               R1_lower_limit=0.3, R1_upper_limit=3.0,
                               BP_lower_limit=-1, BP_upper_limit=15):
    """
    Estimate k2prime from individual voxels within ROIs.
    
    Args:
        pet_data (ndarray): 4D PET data array
        t_tac (ndarray): Time points array
        ref_tac (ndarray): Reference region TAC
        roi_masks (dict): Dictionary of ROI masks
        brain_mask (ndarray): Brain mask array
        multstart_iter (int): Number of iterations for multi-start optimization
        frame_durations (list): How long each time frame lasts
        verbose (bool): Flag to enable detailed output
        k2_prime_threshold (float): Minimum k2prime value to accept
        k2_prime_upper_limit (float): Maximum k2prime value to accept
        R1_lower_limit (float): Minimum R1 value to accept
        R1_upper_limit (float): Maximum R1 value to accept
        BP_lower_limit (float): Minimum BP value to accept  
        BP_upper_limit (float): Maximum BP value to accept
        
    Returns:
        dict: Dictionary containing k2prime and voxel results
    """
    
    if verbose:
        print("\nFirst pass: Estimating k2prime from individual voxels...")
        print(f"  k2prime range: [{k2_prime_threshold:.3f}, {k2_prime_upper_limit:.3f}]")
        print(f"  R1 range: [{R1_lower_limit:.3f}, {R1_upper_limit:.3f}]")
        print(f"  BP range: [{BP_lower_limit:.1f}, {BP_upper_limit:.1f}]")
    
    k2prime_values = []
    roi_summary = {}
    total_voxels_processed = 0
    fail_counter = 0
    reject_counter = 0
    
    # Initialize ROI-specific counters
    for roi_name in roi_masks.keys():
        roi_summary[roi_name] = {
            'total_voxels': np.sum(roi_masks[roi_name]),
            'valid_voxels': 0,
            'failed_voxels': 0,
            'rejected_voxels': 0,
            'k2prime_values': []
        }
    
    total_voxels = sum(roi_summary[roi]['total_voxels'] for roi in roi_summary.keys())
    
    if verbose:
        print(f"Processing {total_voxels} voxels across all ROIs")
    
    # Loop through each ROI and process its voxels
    for roi_name, roi_mask in roi_masks.items():
        if verbose:
            print(f"  Processing ROI: {roi_name}")
        
        shape = pet_data.shape[:3]
        roi_voxel_counter = 0
        
        # Loop through voxels using the same structure as create_parametric_images
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if roi_mask[x, y, z]:
                        total_voxels_processed += 1
                        roi_voxel_counter += 1
                        
                        # Show progress periodically
                        if verbose and total_voxels_processed % 1000 == 0:
                            progress = 100 * total_voxels_processed / total_voxels
                            fail_rate = 100 * fail_counter / total_voxels_processed if total_voxels_processed > 0 else 0
                            reject_rate = 100 * reject_counter / total_voxels_processed if total_voxels_processed > 0 else 0
                            print(f"  Progress: {total_voxels_processed}/{total_voxels} ({progress:.1f}%)")
                            print(f"    Failed fits: {fail_counter} ({fail_rate:.1f}%)")
                            print(f"    Rejected fits: {reject_counter} ({reject_rate:.1f}%)")
                        
                        # Extract voxel TAC
                        voxel_tac = pet_data[x, y, z, :]
                        
                        # Pre-screen voxels with very low activity
                        mean_activity = np.mean(voxel_tac)
                        ref_activity = np.mean(ref_tac)
                        if mean_activity < 0.1 * ref_activity:
                            fail_counter += 1
                            roi_summary[roi_name]['failed_voxels'] += 1
                            continue
                        
                        try:
                            weights = calculate_weights(frame_durations, voxel_tac) if frame_durations is not None else np.ones_like(t_tac)
                            
                            # Use region-specific bounds if available
                            bounds = region_bounds.get(roi_name, default_bounds)
                            
                            result = srtm2(
                                t_tac=t_tac,
                                reftac=ref_tac,
                                roitac=voxel_tac,
                                k2prime=None,  # Estimate k2prime
                                weights=weights,
                                multstart_iter=min(multstart_iter, 10),  # Limit iterations for speed
                                printvals=False,
                                R1_start=bounds["R1"]["start"], 
                                R1_lower=bounds["R1"]["lower"], 
                                R1_upper=bounds["R1"]["upper"],
                                k2prime_start=bounds["k2prime"]["start"], 
                                k2prime_lower=bounds["k2prime"]["lower"], 
                                k2prime_upper=bounds["k2prime"]["upper"],
                                bp_start=bounds["bp"]["start"], 
                                bp_lower=bounds["bp"]["lower"], 
                                bp_upper=bounds["bp"]["upper"]
                            )
                            
                            k2prime_value = result['par']['k2prime'].values[0]
                            R1_value = result['par']['R1'].values[0]
                            bp_value = result['par']['bp'].values[0]
                            
                            # Quality control checks
                            k2prime_valid = True
                            
                            # Check parameter ranges
                            if not (k2_prime_threshold <= k2prime_value <= k2_prime_upper_limit):
                                k2prime_valid = False
                            if not (R1_lower_limit <= R1_value <= R1_upper_limit):
                                k2prime_valid = False
                            if not (BP_lower_limit <= bp_value <= BP_upper_limit):
                                k2prime_valid = False
                            if k2prime_value <= 0:
                                k2prime_valid = False
                            
                            if k2prime_valid:
                                k2prime_values.append(k2prime_value)
                                roi_summary[roi_name]['valid_voxels'] += 1
                                roi_summary[roi_name]['k2prime_values'].append(k2prime_value)
                            else:
                                reject_counter += 1
                                roi_summary[roi_name]['rejected_voxels'] += 1
                                
                        except Exception as e:
                            fail_counter += 1
                            roi_summary[roi_name]['failed_voxels'] += 1
                            if verbose and fail_counter % 1000 == 0:
                                print(f"    Fitting error at ({x},{y},{z}): {str(e)[:50]}...")
                            continue
    
    # Calculate ROI-specific statistics
    for roi_name in roi_summary.keys():
        roi_k2prime_values = roi_summary[roi_name]['k2prime_values']
        roi_summary[roi_name]['success_rate'] = (
            roi_summary[roi_name]['valid_voxels'] / roi_summary[roi_name]['total_voxels'] * 100 
            if roi_summary[roi_name]['total_voxels'] > 0 else 0
        )
        roi_summary[roi_name]['k2prime_mean'] = np.mean(roi_k2prime_values) if roi_k2prime_values else 0
        roi_summary[roi_name]['k2prime_std'] = np.std(roi_k2prime_values) if roi_k2prime_values else 0
        
        if verbose:
            print(f"  ROI {roi_name}: {roi_summary[roi_name]['valid_voxels']} valid / "
                  f"{roi_summary[roi_name]['total_voxels']} total "
                  f"({roi_summary[roi_name]['success_rate']:.1f}% success)")
            if roi_k2prime_values:
                print(f"    k2prime: {roi_summary[roi_name]['k2prime_mean']:.4f} ± "
                      f"{roi_summary[roi_name]['k2prime_std']:.4f}")
    
    # Check if we have any valid estimates
    if len(k2prime_values) == 0:
        raise ValueError("No valid k2prime estimates obtained from any voxel!")
    
    # Calculate global k2prime
    global_k2prime = np.median(k2prime_values)
    
    # Final validation and clamping
    if not (0.001 <= global_k2prime <= 1.0):
        if verbose:
            print(f":warning: Warning: Global k2prime {global_k2prime:.4f} is outside expected range [0.001, 1.0]")
        global_k2prime = np.clip(global_k2prime, 0.001, 1.0)
        if verbose:
            print(f"Clamped to: {global_k2prime:.4f}")
    
    
    # Calculate ROI-specific statistics
    for roi_name in roi_summary.keys():
        roi_k2prime_values = roi_summary[roi_name]['k2prime_values']
        roi_summary[roi_name]['success_rate'] = (
            roi_summary[roi_name]['valid_voxels'] / roi_summary[roi_name]['total_voxels'] * 100 
            if roi_summary[roi_name]['total_voxels'] > 0 else 0
        )
        roi_summary[roi_name]['k2prime_mean'] = np.mean(roi_k2prime_values) if roi_k2prime_values else 0
        roi_summary[roi_name]['k2prime_std'] = np.std(roi_k2prime_values) if roi_k2prime_values else 0
        
        if verbose:
            print(f"  ROI {roi_name}: {roi_summary[roi_name]['valid_voxels']} valid / "
                  f"{roi_summary[roi_name]['total_voxels']} total "
                  f"({roi_summary[roi_name]['success_rate']:.1f}% success)")
            if roi_k2prime_values:
                print(f"    k2prime: {roi_summary[roi_name]['k2prime_mean']:.4f} ± "
                      f"{roi_summary[roi_name]['k2prime_std']:.4f}")
    
    # Check if we have any valid estimates
    if len(k2prime_values) == 0:
        raise ValueError("No valid k2prime estimates obtained from any voxel!")
    
    # Calculate global k2prime
    global_k2prime = np.median(k2prime_values)
    
    # Final validation and clamping
    if not (0.001 <= global_k2prime <= 1.0):
        if verbose:
            print(f":warning: Warning: Global k2prime {global_k2prime:.4f} is outside expected range [0.001, 1.0]")
        global_k2prime = np.clip(global_k2prime, 0.001, 1.0)
        if verbose:
            print(f"Clamped to: {global_k2prime:.4f}")

    # Summary output
    if verbose:
        print(f"\n:bar_chart: Voxel-wise k2prime estimation summary:")
        print(f"  Total voxels processed: {total_voxels_processed}")
        print(f"  Valid voxels: {len(k2prime_values)}")
        print(f"  Overall success rate: {len(k2prime_values)/total_voxels_processed*100:.1f}%")
        print(f"  k2prime range: [{np.min(k2prime_values):.4f}, {np.max(k2prime_values):.4f}]")
        print(f"  k2prime mean: {np.mean(k2prime_values):.4f}")
        print(f"  k2prime median: {global_k2prime:.4f}")
        print(f"  k2prime std: {np.std(k2prime_values):.4f}")
        print(f"  k2prime CV: {np.std(k2prime_values)/np.mean(k2prime_values)*100:.1f}%")
    
    return {
        'global_k2prime': global_k2prime,
        'k2prime_values': k2prime_values,
        'roi_results': roi_summary,
        'k2prime_mean': np.mean(k2prime_values),
        'k2prime_std': np.std(k2prime_values),
        'k2prime_cv': np.std(k2prime_values)/np.mean(k2prime_values)*100,
        'success_rate': len(k2prime_values)/total_voxels_processed*100
    }

def estimate_k2prime(t_tac, ref_tac, roi_tacs, frame_durations,
                      multstart_iter=100, verbose=True,
                      k2_prime_threshold=0.03, k2_prime_upper_limit=0.4,
                      R1_lower_limit=0.3, R1_upper_limit=3.0,
                      BP_lower_limit=-1, BP_upper_limit=15):
    """
    First pass: Estimate k2prime from ROIs using conventional SRTM.
    Uses simple median instead of weighted median.
    
    Args:
        t_tac (ndarray): Time points array
        ref_tac (ndarray): Reference region TAC
        roi_tacs (dict): Dictionary of ROI TACs
        frame_durations (list): How long each time frame lasts
        multstart_iter (int): Number of iterations for multi-start optimization
        verbose (bool): Flag to enable detailed output
        k2_prime_threshold (float): Minimum k2prime value to accept
        k2_prime_upper_limit (float): Maximum k2prime value to accept
        R1_lower_limit (float): Minimum R1 value to accept
        R1_upper_limit (float): Maximum R1 value to accept
        BP_lower_limit (float): Minimum BP value to accept  
        BP_upper_limit (float): Maximum BP value to accept
        
    Returns:
        dict: Dictionary containing k2prime and ROI results
    """
    
    if verbose:
        print("\nFirst pass: Estimating k2prime from ROIs...")
        print(f"  k2prime range: [{k2_prime_threshold:.3f}, {k2_prime_upper_limit:.3f}]")
        print(f"  R1 range: [{R1_lower_limit:.3f}, {R1_upper_limit:.3f}]")
        print(f"  BP range: [{BP_lower_limit:.1f}, {BP_upper_limit:.1f}]")
        print(f"  Using simple median (no weighting)")
    
    roi_results = {}
    k2prime_values = []
    
    # Fit all ROIs
    for roi_name, roi_tac in roi_tacs.items():
        if verbose:
            print(f"  Fitting ROI: {roi_name}")
        
        bounds = region_bounds.get(roi_name, default_bounds)
        
        try:
            weights = calculate_weights(frame_durations, roi_tac) if frame_durations is not None else np.ones_like(t_tac)
            
            result = srtm2(
                t_tac=t_tac,
                reftac=ref_tac,
                roitac=roi_tac,
                k2prime=None,  # Estimate k2prime
                weights=weights,
                multstart_iter=multstart_iter,
                printvals=False,  # Keep individual fits quiet
                R1_start=bounds["R1"]["start"], 
                R1_lower=bounds["R1"]["lower"], 
                R1_upper=bounds["R1"]["upper"],
                k2prime_start=bounds["k2prime"]["start"], 
                k2prime_lower=bounds["k2prime"]["lower"], 
                k2prime_upper=bounds["k2prime"]["upper"],
                bp_start=bounds["bp"]["start"], 
                bp_lower=bounds["bp"]["lower"], 
                bp_upper=bounds["bp"]["upper"]
            )
            
            k2prime_value = result['par']['k2prime'].values[0]
            R1_value = result['par']['R1'].values[0]
            bp_value = result['par']['bp'].values[0]
            
            # Quality control checks
            k2prime_valid = True
            reasons = []
            
            # Check k2prime range
            if not (k2_prime_threshold <= k2prime_value <= k2_prime_upper_limit):
                k2prime_valid = False
                reasons.append(f"k2prime out of range: {k2prime_value:.4f} "
                              f"(expected: {k2_prime_threshold:.3f}-{k2_prime_upper_limit:.3f})")
            
            # Check R1 range
            if not (R1_lower_limit <= R1_value <= R1_upper_limit):
                k2prime_valid = False
                reasons.append(f"R1 out of range: {R1_value:.4f} "
                              f"(expected: {R1_lower_limit:.3f}-{R1_upper_limit:.3f})")
            
            # Check BP range
            if not (BP_lower_limit <= bp_value <= BP_upper_limit):
                k2prime_valid = False
                reasons.append(f"BP out of range: {bp_value:.4f} "
                              f"(expected: {BP_lower_limit:.1f}-{BP_upper_limit:.1f})")
            
            # Additional sanity checks
            if k2prime_value <= 0:
                k2prime_valid = False
                reasons.append(f"k2prime non-positive: {k2prime_value:.4f}")
            
            if k2prime_valid:
                roi_results[roi_name] = result
                k2prime_values.append(k2prime_value)
                
                if verbose:
                    print(f"    :white_check_mark:[bold green] Accepted: R1={R1_value:.3f}, k2prime={k2prime_value:.4f}, BP={bp_value:.3f}")
            else:
                if verbose:
                    print(f"    :x:[bold red] Rejected: {'; '.join(reasons)}")
                    
        except Exception as e:
            if verbose:
                print(f"    :x:[bold red] Fitting failed: {e}")
            continue
    
    # Check if we have any valid estimates
    if len(k2prime_values) == 0:
        raise ValueError("No valid k2prime estimates obtained from any ROI!")
    
    # Calculate simple median (no weighting)
    global_k2prime = np.median(k2prime_values)
    
    # Final validation and clamping
    if not (0.001 <= global_k2prime <= 1.0):
        if verbose:
            print(f":warning: Warning: Global k2prime {global_k2prime:.4f} is outside expected range [0.001, 1.0]")
        global_k2prime = np.clip(global_k2prime, 0.001, 1.0)
        if verbose:
            print(f"Clamped to: {global_k2prime:.4f}")
    
    # Summary output
    if verbose:
        print(f"\n:bar_chart: k2prime estimation summary:")
        print(f"  Total ROIs processed: {len(roi_tacs)}")
        print(f"  Valid ROIs: {len(k2prime_values)}")
        print(f"  Success rate: {len(k2prime_values)/len(roi_tacs)*100:.1f}%")
        print(f"  k2prime values: {[f'{k:.4f}' for k in k2prime_values]}")
        print(f"  k2prime range: [{np.min(k2prime_values):.4f}, {np.max(k2prime_values):.4f}]")
        print(f"  k2prime mean: {np.mean(k2prime_values):.4f}")
        print(f"  k2prime median: {global_k2prime:.4f}")
        print(f"  k2prime std: {np.std(k2prime_values):.4f}")
        print(f"  k2prime CV: {np.std(k2prime_values)/np.mean(k2prime_values)*100:.1f}%")
        
        # Quality assessment
        cv = np.std(k2prime_values)/np.mean(k2prime_values)*100
        if cv < 15:
            print(f"  Quality: Excellent consistency (CV < 15%)")
        elif cv < 30:
            print(f"  Quality: Good consistency (CV < 30%)")
        else:
            print(f"  Quality: Poor consistency (CV ≥ 30%) - consider reviewing ROI selection")
    
    return {
        'global_k2prime': global_k2prime,
        'roi_results': roi_results,
        'k2prime_values': k2prime_values,
        'k2prime_mean': np.mean(k2prime_values),
        'k2prime_std': np.std(k2prime_values),
        'k2prime_cv': np.std(k2prime_values)/np.mean(k2prime_values)*100,
        'success_rate': len(k2prime_values)/len(roi_tacs)*100
    }

