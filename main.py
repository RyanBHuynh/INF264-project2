# INF264 Project 2
# Phillip Lei and Ryan Huynh

# TODO:
# - Create neural network
# - Create support vector machine
# - Create decision tree
# - Create model selection feature
# - Perform model analysis
# - Create model visualizations
# - Create automated test pipeline

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Functions
def create_train_test_valid_sets(X, y, seed):
    """
    Parameters:
     - X and y are the input dataset
     - seed: an integer for the randomization seed to ensure reproducibility
    """
    # Create training set
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Split test into test and validation set
    X_test, y_test, X_valid, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    return X_train, y_train, X_test, y_test, X_valid, y_valid

def create_nn():
    """
    Creates a neural network model
    """
    pass

def create_svm():
    """
    Creates a support vector machine model
    """
    pass

def create_decision_tree():
    """
    Creates a decision tree
    """
    pass

def choose_best_model():
    """
    Selects the best model out of the three
    """
    pass

def visualize_image(X_entry):
    """
    Visualizes an image from the X array
    
    Parameters:
     - X_entry is an entry from the array X
    """
    plt.imshow(X_entry.reshape(20,20), vmin=0, vmax=255, cmap="gray")
    plt.show()

# Main
if __name__ == "__main__":
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")