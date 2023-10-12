# INF264 Project 2
# Phillip Lei and Ryan Huynh

# TODO:
# - Create 3 different models to run
# - Create model selection feature
# - Perform model analysis
# - Create model visualizations
# - Create automated test pipeline

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Functions
def visualize_image(X_entry):
    """
    Visualizes an image from the X array
    
    Parameters:
     - X_entry is an entry from the array X
    """
    plt.imshow(X_entry.reshape(20,20), vmin=0, vmax=255, cmap="gray")
    plt.show()

def create_nn(X_train, y_train): 
    pass

def create_dtree(X_train, y_train):
    pass

def create_svm(X_train, y_train):
    pass

def evaluate_model():
    pass

def select_best_model():
    pass

# Main
if __name__ == "__main__":
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")