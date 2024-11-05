# VolSplinesLib

**VolSplinesLib** is a Python library for interpolating implied volatility surfaces using various volatility models. The library provides tools for fitting and interpolating models to market data, supporting popular methods like RFV, SLV, SABR, and SVI.

You can find the library on [PyPI](https://pypi.org/project/VolSplinesLib/) and on [GitHub](https://github.com/hedge0/VolSplinesLib)

## Installation

You can install VolSplinesLib from PyPI:

```bash
pip install VolSplinesLib
```

Or, install directly from the GitHub repository:

```bash
pip install git+https://github.com/hedge0/VolSplinesLib.git
```

## Usage

After installation, you can perform interpolations using the library. Here's a quick example:

```python
import numpy as np
import matplotlib.pyplot as plt
from VolSplinesLib import perform_interpolation

# Sample data
x = np.array([100, 105, 110, 115, 120], dtype=np.float64)  # Strike prices
y_mid = np.array([0.2, 0.18, 0.16, 0.15, 0.14], dtype=np.float64)  # Mid implied volatilities
y_bid = np.array([0.19, 0.17, 0.15, 0.14, 0.13], dtype=np.float64)  # Bid IVs
y_ask = np.array([0.21, 0.19, 0.17, 0.16, 0.15], dtype=np.float64)  # Ask IVs

# Select the model you want to fit ('RFV', 'SLV', 'SABR', 'SVI')
selected_model = 'RFV'

# Perform interpolation
fine_x, interpolated_y = perform_interpolation(x, y_mid, y_bid, y_ask, selected_model)

# Optionally, plot the original data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(x, y_mid, color='blue', label='Market Data')
plt.plot(fine_x, interpolated_y, color='red', label='Fitted {} Model'.format(selected_model))
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.title('{} Model Fitting'.format(selected_model))
plt.legend()
plt.grid(True)
plt.show()
```

## Available Models

1. **RFV (Rational Function Volatility)**: A rational function model for implied volatilities.
2. **SLV (Spline Log Volatility)**: Uses a spline function for implied volatilities.
3. **SABR (Stochastic Alpha Beta Rho)**: A stochastic volatility model.
4. **SVI (Stochastic Volatility Inspired)**: An inspired model for capturing the volatility smile.

## Contributing

Contributions to **VolSplinesLib** are welcome! If you find a bug or have suggestions for improvements, please open an issue or submit a pull request. Make sure to follow these guidelines:

1. Fork the repository and clone it locally.
2. Create a new branch for your feature or fix.
3. Add your changes and include tests for any new functionality.
4. Run the test suite to ensure all tests pass.
5. Submit a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, feel free to reach out to the author via GitHub: [hedge0](https://github.com/hedge0).
