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
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
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

"""

"""
def preprocess_data(X, y):
    pass
    
"""
Responsible for creating the three candidate models: nn, decision tree, support vector machine.

Output: returns array of the three models
"""
def create_all_models():
    pass

"""
Creates one of the three candidates: nn

Output: neural network model - sklearn
"""
def create_nn(X_train, y_train): 

    pass

"""
Creates one of the three candidates: decision tree

Output: decision tree model - sklearn
"""
def create_dtree(X_train, y_train):
    pass

"""
Creates one of the three candidates: support vector machine

Output: support vector machine model - sklearn
"""
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
    
    # Step 1. Preprocess Data, do we need this? Eg: making images smaller
    
    # Note: fix a seed for reproducibility
    
    # Step 2. Split processed data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    # Step 3. Create 3 test models
    
    # Step 4. Select model
    
    # Step 5. Evaluate model
    
    # Step 6. Print out numbers like accuracy. Should be reproducible across runs