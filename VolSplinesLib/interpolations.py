import numpy as np
from VolSplinesLib.minimize import minimize

class Interpolations:
    """
    A class containing methods for fitting different volatility models to market data.

    This class provides static methods for various volatility models and an objective function
    to fit these models to market data using optimization techniques.
    """

    @staticmethod
    def objective_function(params, k, y_mid, y_bid, y_ask, model):
        """
        Compute the weighted sum of squared residuals between the model predictions and market data.

        Parameters:
        - params (np.ndarray): Array of parameters for the volatility model.
        - k (np.ndarray): Log-strike values, i.e., the natural logarithm of the strike prices.
        - y_mid (np.ndarray): Observed mid implied volatilities from market data.
        - y_bid (np.ndarray): Observed bid implied volatilities from market data.
        - y_ask (np.ndarray): Observed ask implied volatilities from market data.
        - model (callable): The volatility model function that predicts implied volatilities.

        Returns:
        - float: The weighted sum of squared residuals between model predictions and market data.
        """
        spread = np.subtract(y_ask, y_bid)
        epsilon = 1e-8
        weights = 1 / (spread + epsilon)
        model_values = model(k, params)
        residuals = model_values - y_mid
        residuals_squared = residuals ** 2
        weighted_residuals = weights * residuals_squared
        return np.sum(weighted_residuals)

    @staticmethod
    def rfv_model(k, params):
        """
        Rational Function Volatility (RFV) model function.

        Mathematical Form:
            IV(k) = [a + b * k + c * k^2] / [1 + d * k + e * k^2]

        Parameters:
        - k (np.ndarray): Log-strike values.
        - params (np.ndarray): Array of parameters [a, b, c, d, e].

        Returns:
        - np.ndarray: Implied volatilities predicted by the RFV model.
        """
        a, b, c, d, e = params
        numerator = a + b * k + c * k ** 2
        denominator = 1 + d * k + e * k ** 2
        return numerator / denominator

    @staticmethod
    def slv_model(k, params):
        """
        Spline Log Volatility (SLV) model function.

        Mathematical Form:
            IV(k) = a + b * k + c * k^2 + d * k^3 + e * k^4

        Parameters:
        - k (np.ndarray): Log-strike values.
        - params (np.ndarray): Array of parameters [a, b, c, d, e].

        Returns:
        - np.ndarray: Implied volatilities predicted by the SLV model.
        """
        a, b, c, d, e = params
        return a + b * k + c * k ** 2 + d * k ** 3 + e * k ** 4

    @staticmethod
    def sabr_model(k, params):
        """
        SABR (Stochastic Alpha Beta Rho) model function.

        Mathematical Form (Approximation):
            IV(k) = α * [1 + β * k + ρ * k^2 + ν * k^3 + f₀ * k^4]

        Parameters:
        - k (np.ndarray): Log-strike values.
        - params (np.ndarray): Array of parameters [α (alpha), β (beta), ρ (rho), ν (nu), f₀ (f0)].

        Returns:
        - np.ndarray: Implied volatilities predicted by the SABR model.
        """
        alpha, beta, rho, nu, f0 = params
        return alpha * (1 + beta * k + rho * k ** 2 + nu * k ** 3 + f0 * k ** 4)

    @staticmethod
    def svi_model(k, params):
        """
        Stochastic Volatility Inspired (SVI) model function.

        Mathematical Form:
            IV(k) = a + b * [ρ * (k - m) + sqrt((k - m)^2 + σ^2)]

        Parameters:
        - k (np.ndarray): Log-strike values.
        - params (np.ndarray): Array of parameters [a, b, ρ (rho), m, σ (sigma)].

        Returns:
        - np.ndarray: Implied volatilities predicted by the SVI model.
        """
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

    @staticmethod
    def fit_model(x, y_mid, y_bid, y_ask, selectedModel):
        """
        Fit the selected volatility model to the market data.

        Parameters:
        - x (np.ndarray): Strike prices of the options.
        - y_mid (np.ndarray): Observed mid implied volatilities from market data.
        - y_bid (np.ndarray): Observed bid implied volatilities from market data.
        - y_ask (np.ndarray): Observed ask implied volatilities from market data.
        - selectedModel (str): The chosen model for fitting. Options are:
            - 'RFV' for Rational Function Volatility model.
            - 'SLV' for Spline Log Volatility model.
            - 'SABR' for SABR model.
            - 'SVI' for Stochastic Volatility Inspired model.

        Returns:
        - np.ndarray: The optimized parameters for the selected volatility model.

        Raises:
        - ValueError: If an invalid model name is provided.
        """
        k = np.log(x)
        initial_guess = [0.2, 0.3, 0.1, 0.2, 0.1]
        bounds = [(-np.inf, np.inf)] * 5
        try:
            model = {
                'RFV': Interpolations.rfv_model,
                'SLV': Interpolations.slv_model,
                'SABR': Interpolations.sabr_model,
                'SVI': Interpolations.svi_model,
            }[selectedModel]
        except KeyError:
            raise ValueError(f"Invalid model selected. Available models: 'RFV', 'SLV', 'SABR', 'SVI'.")

        def funcGrad(params):
            """
            Compute the objective function value and its gradient at the given parameters.

            Parameters:
            - params (np.ndarray): Current estimate of the model parameters.

            Returns:
            - tuple:
                - f (float): Current value of the objective function.
                - grad (np.ndarray): Gradient of the objective function with respect to the parameters.
            """
            f = Interpolations.objective_function(params, k, y_mid, y_bid, y_ask, model)
            epsilon = 1e-8
            n_params = len(params)
            grad = np.zeros(n_params)
            for i in range(n_params):
                params_eps = params.copy()
                params_eps[i] += epsilon
                f_eps = Interpolations.objective_function(params_eps, k, y_mid, y_bid, y_ask, model)
                grad[i] = (f_eps - f) / epsilon
            return f, grad

        result = minimize(funcGrad, np.array(initial_guess), bounds)

        if result['status'] != 0:
            print("Optimization failed:", result['message'])

        return result['x']
