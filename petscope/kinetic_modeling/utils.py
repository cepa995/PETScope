def fix_multstartpars(start, lower, upper, multstart_iter, multstart_lower=None, multstart_upper=None):
    """
    Check and fix multstart parameters.
    
    Checks missing multstart boundaries, verifies if some parameters should not be iterated,
    and ensures correct lengths for multstart bounds.
    
    Parameters:
    start : dict
        Original starting values without multstart.
    lower : dict
        Lower fitting bounds.
    upper : dict
        Upper fitting bounds.
    multstart_iter : int or list
        Number of iterations.
    multstart_lower : dict, optional
        Lower starting bounds.
    multstart_upper : dict, optional
        Upper starting bounds.
    
    Returns:
    dict
        A dictionary containing adjusted multstart_lower and multstart_upper bounds.
    """
    if isinstance(multstart_iter, int) or len(multstart_iter) == len(start):
        multstart_l = lower.copy()
        multstart_u = upper.copy()
        
        parnames = start.keys()
        
        # Adding multstart boundaries
        if multstart_lower is not None:
            if not isinstance(multstart_lower, dict) or not all(k in parnames for k in multstart_lower.keys()):
                raise ValueError("multstart_lower should be a dictionary with keys matching the parameters to be fitted")
            multstart_lower = {**multstart_l, **multstart_lower}
        else:
            multstart_lower = multstart_l
        
        if multstart_upper is not None:
            if not isinstance(multstart_upper, dict) or not all(k in parnames for k in multstart_upper.keys()):
                raise ValueError("multstart_upper should be a dictionary with keys matching the parameters to be fitted")
            multstart_upper = {**multstart_u, **multstart_upper}
        else:
            multstart_upper = multstart_u
        
        # No multstart for some variables
        if isinstance(multstart_iter, list) and any(i == 1 for i in multstart_iter):
            non_iterable = [i for i, val in enumerate(multstart_iter) if val == 1]
            for idx in non_iterable:
                key = list(start.keys())[idx]
                multstart_lower[key] = start[key]
                multstart_upper[key] = start[key]
        
        # Ensure correct order
        multstart_lower = {k: multstart_lower[k] for k in start.keys()}
        multstart_upper = {k: multstart_upper[k] for k in start.keys()}
        
        return {
            "multstart_lower": multstart_lower,
            "multstart_upper": multstart_upper
        }
    else:
        raise ValueError("multstart_iter should be of length 1 or the same length as the number of parameters")
