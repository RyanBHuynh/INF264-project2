# INF264 Project 2
# Phillip Lei and Ryan Huynh

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Functions
def visualize_image(X_entry):
    plt.imshow(X_entry.reshape(20,20), vmin=0, vmax=255, cmap="gray")
    plt.show()

# Main
if __name__ == "__main__":
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")

    y.head()
    visualize_image(X[0])