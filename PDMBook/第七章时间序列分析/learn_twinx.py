import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x)

# Create a figure and axes object
fig, ax1 = plt.subplots()

# Plot the first dataset on the left y-axis
ax1.plot(x, y1, color='blue')
ax1.set_ylabel('y1', color='blue')

# Create a second y-axis on the right side of the plot
ax2 = ax1.twinx()

# Plot the second dataset on the right y-axis
ax2.plot(x, y2, color='red')
ax2.set_ylabel('y2', color='red')

# Show the plot
plt.show()
