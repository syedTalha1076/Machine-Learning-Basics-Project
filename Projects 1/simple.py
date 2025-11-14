import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: User inputs the number of data points
n = int(input("How many data points do you want to enter? "))

# Step 2: Input feature (X) and target (Y)
x_data = []
y_data = []

print("\nEnter your data (Example: Hours -> Score):")
for i in range(n):
    x_val = float(input(f"Enter value for X (e.g., Hours) [{i+1}/{n}]: "))
    y_val = float(input(f"Enter corresponding Y (e.g., Score) [{i+1}/{n}]: "))
    x_data.append(x_val)
    y_data.append(y_val)

# Step 3: Create DataFrame
df = pd.DataFrame({'X': x_data, 'Y': y_data})

# Step 4: Prepare data
x = df['X'].values.reshape(-1, 1)
y = df['Y'].values.reshape(-1, 1)

# Step 5: Train model
reg = LinearRegression()
reg.fit(x, y)

# Step 6: Prediction for a new value
x_test = float(input("\nEnter a new value of X to predict Y: "))
predicted_y = reg.predict([[x_test]])
print(f"Predicted Y for X = {x_test} is: {predicted_y[0][0]:.2f}")

# Step 7: Plotting
plt.figure(figsize=(6,4))
plt.scatter(x, y, color='blue', label='User Data')
plt.plot(x, reg.predict(x), color='red', label='Regression Line')
plt.scatter(x_test, predicted_y, color='green', s=100, label='Predicted Point')
plt.xlabel("X (e.g., Hours)")
plt.ylabel("Y (e.g., Score)")
plt.title("Linear Regression on User Input")
plt.legend()
plt.grid(True)
plt.show()
