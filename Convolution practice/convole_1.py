import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Example 2D input array (e.g., a simple image)
image = np.array([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

# Example 2D kernel (e.g., an edge-detection kernel)
kernel = np.array([[1, 0, -1], 
                   [1, 0, -1], 
                   [1, 0, -1]])

# Perform 2D convolution
output = signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)

# Print the result
print("Convolved Output:\n", output)

# Visualize the input and output
plt.figure(figsize=(6, 6))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Convolved Image")
plt.imshow(output, cmap='gray', interpolation='nearest')
plt.colorbar()

plt.show()
