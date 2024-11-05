import numpy as np
import matplotlib.pyplot as plt
from VolSplinesLib.interpolation_function import perform_interpolation  # Import the function

# Sample data (replace with your actual market data)
x = np.array([100, 105, 110, 115, 120], dtype=np.float64)  # Strikes
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
