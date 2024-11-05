import numpy as np
from VolSplinesLib.interpolations import Interpolations

def perform_interpolation(x, y_mid, y_bid, y_ask, selected_model):
    """
    Perform interpolation using the specified model.

    Args:
        x (np.ndarray): Strike prices.
        y_mid (np.ndarray): Mid implied volatilities.
        y_bid (np.ndarray): Bid implied volatilities.
        y_ask (np.ndarray): Ask implied volatilities.
        selected_model (str): Model to use for interpolation ('RFV', 'SLV', 'SABR', 'SVI').

    Returns:
        tuple: fine_x (np.ndarray), interpolated_y (np.ndarray)
    """
    x_min = np.min(x)
    x_max = np.max(x)
    x_normalized = (x - x_min) / (x_max - x_min) + 0.5

    params = Interpolations.fit_model(x_normalized, y_mid, y_bid, y_ask, selected_model)

    fine_x_min = np.min(x_normalized)
    fine_x_max = np.max(x_normalized)
    fine_x_normalized = np.linspace(fine_x_min, fine_x_max, 800)

    model_function = {
        'RFV': Interpolations.rfv_model,
        'SLV': Interpolations.slv_model,
        'SABR': Interpolations.sabr_model,
        'SVI': Interpolations.svi_model,
    }.get(selected_model)

    if model_function:
        interpolated_y = model_function(np.log(fine_x_normalized), params)
        fine_x = x_min + (fine_x_normalized - 0.5) * (x_max - x_min)
        return fine_x, interpolated_y
    else:
        raise ValueError("Model function is undefined.")
