import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Generate synthetic data
np.random.seed(42)  # For reproducibility
m_true = 2.5  # True slope
c_true = 5.0  # True intercept
num_points = 100  # Number of data points

#Generate x values (compare results with and without normalization)
x = np.linspace(0, 100, num_points).reshape(-1, 1) 
x = (x - x.min()) / (x.max() - x.min()) # Normalize x values between 0 and 1 for better NN performance

# Generate noisy data (linear or quadratic)
noise = np.random.normal(0, 0.2, size=(num_points, 1))  # Adding noise
#y = m_true * x + c_true + noise  #Linear
y = m_true * x + c_true + 0.5 * x**2 + noise  # Quadratic 

# Generate original (noise-free) quadratic function
y_true = m_true * x + c_true + 0.5 * x**2 

# Fit linear regression model
model_lr = LinearRegression()
model_lr.fit(x, y)

# Predictions for linear regression
y_pred_lr = model_lr.predict(x)

# Neural network model
model_nn = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),  # First hidden layer with 128 neurons
    layers.Dense(128, activation='relu'), # Second hidden layer with 128 neurons
    layers.Dense(64, activation='relu'), # Third hidden layer with 64 neurons
    layers.Dense(1) # Output layer with a single neuron
])

# Compile the neural network with Adam optimizer and Mean Squared Error loss
model_nn.compile(optimizer='adam', loss='mse')

# Train neural network with 200 epochs
model_nn.fit(x, y, epochs=200, verbose=0)

# Predictions for neural network
y_pred_nn = model_nn.predict(x)

# Plot results
plt.scatter(x, y, label="Data", alpha=0.6)
plt.plot(x, y_true, color='green', label="True Function", linestyle='dotted')  # Original function without noise
plt.plot(x, y_pred_lr, color='red', label=f"Linear Regression: y = {model_lr.coef_[0][0]:.2f}x + {model_lr.intercept_[0]:.2f}")
plt.plot(x, y_pred_nn, color='blue', label="Neural Network Prediction", linestyle='dashed')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Comparison: Linear Regression vs Neural Network")
plt.show()

# Print learned parameters
print(f"Estimated slope (m) from Linear Regression: {model_lr.coef_[0][0]:.2f}")
print(f"Estimated intercept (c) from Linear Regression: {model_lr.intercept_[0]:.2f}")

