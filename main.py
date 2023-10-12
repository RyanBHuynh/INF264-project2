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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Functions
def create_train_test_valid_sets(X, y, seed):
    """
    Parameters:
     - X and y are the input dataset
     - seed: an integer for the randomization seed to ensure reproducibility

    Return value: a dictionary containing train, test, and validation sets for X and y
    """
    # Create training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # Split test into test and validation set
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)

    model_dict = {
        "X": {
            "train": X_train,
            "test": X_test,
            "valid": X_valid
        },
        "y": {
            "train": y_train,
            "test": y_test,
            "valid": y_valid
        }
    }

    return model_dict

def choose_best_model(candidates, X_test, y_test):
    """
    Selects the best model out of the three
    """
    nn = candidates["Neural Network"]
    nn_pred = nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    print("nn_accuracy:", nn_accuracy)

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

Output: returns data structure of the three models
"""
def create_all_models(X_train, y_train, seed):
    # 1: NEURAL NETWORK
    nn = create_nn(X_train, y_train, seed)
    # 2: DECISION TREE
    dtree = create_dtree(X_train, y_train)
    # 3: SUPPORT VECTOR MACHINE
    svm = create_svm(X_train, y_train)
    
    candidates = {
        "Neural Network" : nn,
        "Decision Tree " : dtree,
        "Support Vector Machine" : svm
    } 
    
    return candidates

"""
Creates one of the three candidates: nn

Output: neural network model - sklearn
"""
def create_nn(X_train, y_train, seed): 
    neural_network = MLPClassifier(
        hidden_layer_sizes = (100, 50, 25, 12),
        max_iter = 1000,
        random_state = seed
    )
    trained_nn = neural_network.fit(X_train, y_train)

    return trained_nn

"""
Creates one of the three candidates: decision tree

Output: decision tree model - sklearn
"""
def create_dtree(X_train, y_train):
<<<<<<< HEAD
    dtree = DecisionTreeClassifier(random_state=seed)
    
    parameter_grid = {
        "max_depth": [5, 10, 15, 20, 25, 30],
        "criterion": ["gini", "entropy"],
        "splitter": ["best", "random"]
    }
    
    grid_search = GridSearchCV(dtree, parameter_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_dtree = grid_search.best_estimator_
    
    return best_dtree
=======
    return None
>>>>>>> ecf66956a131a8823e2a16e9052e204ea2d1062b

"""
Creates one of the three candidates: support vector machine

Output: support vector machine model - sklearn
"""
def create_svm(X_train, y_train):
    return None


def evaluate_model():
    pass

# Main
if __name__ == "__main__":
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")
    
    # Step 1. Preprocess Data, do we need this? Eg: making images smaller
    
    # Note: fix a seed for reproducibility
    # seed = 143 # Å
    seed = 10000
    # Step 2. Split processed data
    model_dict = create_train_test_valid_sets(X, y, seed)
    
    # Step 3. Create 3 test models
    X_train = model_dict["X"]["train"]
    y_train = model_dict["y"]["train"]
    models = create_all_models(X_train, y_train, seed)
    
    # Step 4. Select model
    X_test = model_dict['X']['test']
    y_test = model_dict['y']['test']
    choose_best_model(models, X_test, y_test)
    # Step 5. Evaluate model
    
    # Step 6. Print out numbers like accuracy. Should be reproducible across runs