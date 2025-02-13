import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rich import print
from scipy.optimize import least_squares
from petscope.kinetic_modeling.utils import fix_multstartpars


def process_srtm2_output(result, modeldata, upper, lower, k2prime=None):
    """
    Processes the output of SRTM2 model fitting to check parameter limits, 
    compute fitted TACs, and return structured output.

    Arguments:
        result (dict): The output from fit_srtm2_model containing fitted parameters.
        modeldata (dict): Dictionary containing 't_tac', 'roitac', 'reftac'.
        upper (dict): Upper bounds of parameters.
        lower (dict): Lower bounds of parameters.
        k2prime (float, optional): If not None, the k2prime parameter is fixed.

    Returns
        dict :A dictionary containing fitted parameters, standard errors, fitted TACs, 
            and metadata about the model fit.
    """
    fitted_params = np.round(np.array(list(result.values())), 3)
    upper_bounds = np.round(np.array([upper[key] for key in result.keys()]), 3)
    lower_bounds = np.round(np.array([lower[key] for key in result.keys()]), 3)

    # Check if parameters hit upper or lower limits
    limcheck_u = np.equal(fitted_params, upper_bounds)
    limcheck_l = np.equal(fitted_params, lower_bounds)
    limcheck = limcheck_u | limcheck_l

    if np.any(limcheck):
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
        return 0.0  # Replace with real SE calculation if available

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

def kinfit_convolve(a, b, step):
    """
    Convolve two signals a and b using a fixed time step.
    This approximates the continuous convolution using a Riemann sum.
    """
    # The full convolution is computed; we then take only the first len(a) points
    conv = np.convolve(a, b, mode='full')[:len(a)] * step
    return conv

def srtm2_model(t_tac, reftac, R1, k2prime, bp):
    """
    Simplified Reference Tissue Model 2 (SRTM2)

    Arguments:
        t_tac (array_like): 1D array of time points (in minutes). (We use the time at mid‐frame.)
        reftac (array_like): 1D array of radioactivity concentrations in the reference tissue.
        R1 (float): Parameter R1.
        k2prime (float): Parameter k2prime.
        bp (float): Parameter bp.

    Returns:
        outtac (ndarray): 1D array of the predicted target tissue time-activity curve.
    """
    # Reduce points if instability
    interptime = np.linspace(np.min(t_tac), np.max(t_tac), 512)  
    step = interptime[1] - interptime[0]

    if step <= 0 or step < 1e-10:
        raise ValueError("Step size is too small!")

    iref = np.interp(interptime, t_tac, reftac)
    k2a = (R1 * k2prime) / (bp + 1)
    
    a = R1 * (k2prime - k2a) * iref
    b = np.exp(-k2a * interptime)

    ND = R1 * iref
    BOUND = kinfit_convolve(a, b, step)
    i_outtac = ND + BOUND

    outtac = np.interp(t_tac, interptime, i_outtac)
    return outtac

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
    
    if k2prime_fixed is not None:
        # Check if k2prime is outside of the bounds
        if not lower['k2prime'] <= k2prime_fixed <= upper['k2prime']:
            raise ValueError("Fixed k2prime value is outside allowed bounds.")

        # Define a model function with k2prime held fixed.
        def model_func(t, R1, bp):
            return srtm2_model(t, reftac, R1, k2prime_fixed, bp)
        # Set up the starting values and bounds for the two free parameters.
        if isinstance(start, dict):
            p0 = [start['R1'], start['bp']]
            lower_bounds = [lower['R1'], lower['bp']]
            upper_bounds = [upper['R1'], upper['bp']]
        else:
            p0 = start
            lower_bounds = lower
            upper_bounds = upper
    else:
        # All three parameters are free.
        def model_func(t, R1, k2prime, bp):
            return srtm2_model(t, reftac, R1, k2prime, bp)
        if isinstance(start, dict):
            p0 = [start['R1'], start['k2prime'], start['bp']]
            lower_bounds = [lower['R1'], lower['k2prime'], lower['bp']]
            upper_bounds = [upper['R1'], upper['k2prime'], upper['bp']]
        else:
            p0 = start
            lower_bounds = lower
            upper_bounds = upper

    # Define residual function
    def residuals(params, t_tac, roitac):
        R1, k2prime, bp = params
        model_pred = srtm2_model(t_tac, roitac, R1, k2prime, bp)
        residuals_arr = (model_pred - roitac) * weights  # Minimize the difference
        if np.isnan(residuals_arr).any():
            return np.full_like(residuals_arr, np.inf)  # Avoid NaNs in optimization
        return residuals_arr
    
    # Choose between a single fit or a multi-start approach
    if multstart_iter == 1:
        # Perform optimization
        result = least_squares(residuals, p0, args=(t_tac, roitac), bounds=(lower_bounds, upper_bounds))
        optimized_paramters = result.x
    else:
        best_cost = np.inf
        optimized_paramters = None
        for i in range(multstart_iter):
            # Generate a random starting point between the given bounds.
            p0_trial = np.random.uniform(list(multstart_lower.values()), list(multstart_upper.values()))
            try:
                result = least_squares(residuals, p0_trial, args=(t_tac, roitac), bounds=(lower_bounds, upper_bounds))
                optimized_paramters_trial = result.x
                
                # Compute a simple cost (sum of squared residuals)
                cost = np.sum((model_func(t_tac, *optimized_paramters_trial) - roitac) ** 2)
                if printvals:
                    print(f"Iteration {i}: cost = {cost}")
                if cost < best_cost:
                    best_cost = cost
                    optimized_paramters = optimized_paramters_trial
            except RuntimeError:
                if printvals:
                    print(f"Iteration {i}: fit did not converge.")
                continue

    if k2prime_fixed is not None:
        return {'R1': optimized_paramters[0], 'k2prime': k2prime_fixed, 'bp': optimized_paramters[1]}
    else:
        return {'R1': optimized_paramters[0], 'k2prime': optimized_paramters[1], 'bp': optimized_paramters[2]}

def srtm2(t_tac, reftac, roitac, k2prime=None, weights=None, frame_start_end=None,
          R1_start=1, R1_lower=0, R1_upper=10,
          k2prime_start=0.1, k2prime_lower=0.001, k2prime_upper=1,
          bp_start=1.5, bp_lower=0, bp_upper=15,
          multstart_iter=1, multstart_lower=None, multstart_upper=None,
          printvals=False):
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
        start = {'R1': R1_start, 'k2prime': k2prime_start, 'bp': bp_start}
        lower = {'R1': R1_lower, 'k2prime': k2prime_lower, 'bp': bp_lower}
        upper = {'R1': R1_upper, 'k2prime': k2prime_upper, 'bp': bp_upper}

        print("Note: Without specifying a k2prime value for SRTM2, it is effectively "
              "equivalent to the conventional SRTM model. This can be useful for "
              "selecting an appropriate k2prime value, but without re-fitting the "
              "model with a specified k2prime value, the model is not really SRTM2.")
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
                            printvals=True,
                            k2prime_fixed=k2prime)
    output = process_srtm2_output(result, modeldata, upper, lower, k2prime)
    return output

def plot_srtm2fit(srtm2out, roiname=None, refname=None, save_path="srtm2_fit.png"):
    """
    Generate and save a visualization of the SRTM2 model fit.

    Arguments:
        srtm2out (dict): Output dictionary from the SRTM2 fitting procedure.
            Expected keys:
            - 'tacs' (DataFrame): Time-activity curves containing 'Time', 'Reference', 'Target', and 'Target_fitted'.
            - 'weights' (array_like): Weights used during the model fitting.
        roiname (str, optional): Name of the target region (ROI) to label on the plot. Default is "ROI".
        refname (str, optional): Name of the reference region to label on the plot. Default is "Reference".
        save_path (str, optional): File path to save the generated plot. Default is "srtm2_fit.png".

    Returns:
        None: The function saves the plot as a PNG file at the specified `save_path`.

    Notes:
        - The function creates a scatter plot of measured TAC data with point sizes scaled by weights.
        - The fitted model TAC is plotted as a dashed line for comparison.
        - The plot is saved in high resolution (300 DPI).
    """

    # Extract measured and fitted data
    measured = srtm2out["tacs"][["Time", "Reference", "Target"]].copy()
    measured["Weights"] = srtm2out["weights"]

    fitted = srtm2out["tacs"][["Time", "Target_fitted"]].copy()
    fitted["Weights"] = srtm2out["weights"]

    # Set default names if not provided
    roiname = roiname if roiname else "ROI"
    refname = refname if refname else "Reference"

    # Rename columns for clarity
    measured.rename(columns={"Target": f"{roiname}.measured", "Reference": refname}, inplace=True)
    fitted.rename(columns={"Target_fitted": f"{roiname}.fitted"}, inplace=True)

    # Reshape data for plotting
    tidy_measured = measured.melt(id_vars=["Time", "Weights"], var_name="Region", value_name="Radioactivity")
    tidy_fitted = fitted.melt(id_vars=["Time", "Weights"], var_name="Region", value_name="Radioactivity")

    # Define color palette
    colors = sns.color_palette("Set1", n_colors=3)

    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for measured values with weighting size
    sns.scatterplot(
        data=tidy_measured, x="Time", y="Radioactivity", hue="Region", size="Weights",
        sizes=(10, 100), alpha=0.8, edgecolor="black", palette=colors
    )

    # Line plot for fitted values
    sns.lineplot(
        data=tidy_fitted, x="Time", y="Radioactivity", hue="Region", linewidth=2, linestyle="dashed", palette=colors
    )

    # Formatting
    plt.xlabel("Time (min)")
    plt.ylabel("Radioactivity")
    plt.title("SRTM2 Model Fit")
    plt.legend(title="Region", loc="upper right")
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to prevent displaying it

    print(f"Plot saved as: {save_path}")



if __name__ == "__main__":
    # Read sample data into Pandas DataFrame
    simRef_df = pd.read_csv("./test_data/simRef_example.csv")

    # Parse SRTM2 Input arguments
    t_tac = simRef_df['Times'].values
    reftac = simRef_df['Reference'].values
    roitac = simRef_df['ROI1'].values
    weights = simRef_df['Weights'].values

    # Execute SRTM2
    srtm2_output = srtm2(
        t_tac=t_tac,
        reftac=reftac,
        roitac=roitac,
        weights=weights,
        multstart_iter=1
    )

    print(srtm2_output)
    plot_srtm2fit(
        srtm2out=srtm2_output,
        roiname="Hippocampus",
        refname="Cerebellum",
        save_path="srtm2_plot.png"
    )
