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
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as skimg

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score

# Functions
def create_train_test_valid_sets(X, y, seed):
    """
    Parameters:
     - X and y are the input dataset
     - seed: an integer for the randomization seed to ensure reproducibility

    Return value: a dictionary containing train, test, and validation sets for X and y
    """
    # Create training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=seed)

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

    print("Data successfully split into train, test, and validation sets")
    return model_dict

def choose_best_model(candidates, X_test, y_test):
    """
    Selects the best model out of the three
    """
    best_model = None
    best_score = 0
    
    for name in candidates:
        # 10 is a common cross validation fold value
        model = candidates[name]
        mean_score = cross_val_score(model, X_test, y_test, vc=10).mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            
    print("Best model: ", model)
    print("Score of model: ", best_score)
    
    # # Looking at accuracy values 
    # print("Running neural network...")
    # nn = candidates["Neural Network"]
    # nn_pred = nn.predict(X_test)
    # nn_accuracy = accuracy_score(y_test, nn_pred)
    # print("\nnn_accuracy:", nn_accuracy)
    
    # print("Running decision tree...")
    # dtree = candidates["Decision Tree"]
    # dtree_pred = dtree.predict(X_test)
    # dtree_accuracy = accuracy_score(y_test, dtree_pred)
    # print("dtree_accuracy:", dtree_accuracy)

    # print("Running SVM...")
    # svm = candidates["Support Vector Machine"]
    # svm_pred = svm.predict(X_test)
    # svm_accuracy = accuracy_score(y_test, svm_pred)
    # print("svm_accuracy:", svm_accuracy)

def visualize_image(X_entry):
    """
    Visualizes an image from the X array
    
    Parameters:
     - X_entry is an entry from the array X
    """
    plt.imshow(X_entry.reshape(20,20), vmin=0, vmax=255, cmap="gray")
    plt.show()

"""
Process data by setting image pixels to binary values and scaling images down.
Intended to help computation run time.

"""
def preprocess_data(X):
    processed_X = []
    
    for image in X:
        image = normalize_image_rgb(image)
        image = resize_image(image)
        processed_X.append(image)
    
    return processed_X

"""
Shrinking an image due to running time constraints. Smaller images (np arrays) 
should help our training run faster.

Output: resized np array representing an image
"""
def resize_image(image):
    # print("Image Size: ", len(image))
    # turn image into np array
    if isinstance(image, np.ndarray):
        image = np.array(image)
        
    image = image.reshape(20, 20)
    # print("Image: ", image)
    # Max pooling function to reduce image by a factor of 2
    resized_image = skimg.block_reduce(image, (2, 2), np.max) 

    return resized_image.reshape(100)
    
"""
Normalize pixel values of the image to 0 or 1.

Output: image with each value being binary
"""
def normalize_image_rgb(image):
    border = 127
    for i in range(len(image)):
        if image[i] > border:
            image[i] = 1
        else:
            image[i] = 0
            
    return image    

"""
Responsible for creating the three candidate models: nn, decision tree, support vector machine.

Output: returns data structure of the three models
"""
def create_all_models(X_train, y_train, X_test, y_test, seed):
    # 1: NEURAL NETWORK
    nn = create_nn(X_train, y_train, X_test, y_test, seed, True)
    # 2: DECISION TREE
    dtree = create_dtree(X_train, y_train, X_test, y_test, seed, True)
    # 3: SUPPORT VECTOR MACHINE
    svm = create_svm(X_train, y_train, X_test, y_test, seed, True)
    
    candidates = {
        "Neural Network" : nn,
        "Decision Tree" : dtree,
        "Support Vector Machine" : svm
    } 
    
    return candidates

"""
Creates one of the three candidates: nn

Output: neural network model - sklearn
"""
def create_nn(X_train, y_train, X_test, y_test, seed, tuning=False): 
    print("\nCreating neural network...")
    start_time = time.time()

    neural_network = MLPClassifier(random_state = seed)

    if tuning:
        parameter_grid = {
            "activation": ['logistic', 'relu'],
            "solver": ['sgd', 'adam'],
            "learning_rate": ['constant', 'adaptive']
        }
        
        grid_search = GridSearchCV(neural_network, parameter_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Neural Network Parameters: " + str(best_params))
        best_nn = grid_search.best_estimator_
        best_nn = neural_network.fit(X_train, y_train)
    else:
        best_nn = neural_network.fit(X_train, y_train)
    
    total_time = time.time() - start_time
    print("Neural network training complete")
    print(f"Finished in {total_time} seconds")

    # DEBUG ACCURACY
    nn_pred = best_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    print("nn_accuracy:", nn_accuracy)

    return best_nn

"""
Creates one of the three candidates: decision tree

Output: decision tree model - sklearn
"""
def create_dtree(X_train, y_train, X_test, y_test, seed, tuning=False):
    print("\nCreating decision tree...")
    start_time = time.time()

    dtree = DecisionTreeClassifier(random_state=seed)
    
    if tuning: 
        parameter_grid = {
            "max_depth": [20, 25, 30],
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"]
        }
        
        grid_search = GridSearchCV(dtree, parameter_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Decision Tree Parameters: " + str(best_params))
        best_dtree = grid_search.best_estimator_
        best_dtree = dtree.fit(X_train, y_train)

    else:
        best_dtree = dtree.fit(X_train, y_train)

    total_time = time.time() - start_time
    print("Decision tree created")
    print(f"Finished in {total_time} seconds")

    # DEBUG ACCURACY
    dtree_pred = best_dtree.predict(X_test)
    dtree_accuracy = accuracy_score(y_test, dtree_pred)
    print("dtree_accuracy:", dtree_accuracy)

    return best_dtree

"""
Creates one of the three candidates: support vector machine

Output: support vector machine model - sklearn
"""
def create_svm(X_train, y_train, X_test, y_test, seed, tuning=False):
    print("\nCreating SVMs...")
    start_time = time.time()

    svm = SVC(random_state=seed)
    if tuning:
        print("SVM hyperparameter tuning...")
        parameter_grid = {
            'C': [0.1, 1, 10, 100],  
            'gamma': ["auto", "scale"], 
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
    
        grid_search = GridSearchCV(svm, parameter_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best SVM Parameters: " + str(best_params))
        best_svm = grid_search.best_estimator_

    else:
        best_svm = svm.fit(X_train, y_train)

    total_time = time.time() - start_time
    print("SVM created")
    print(f"Finished in {total_time} seconds")

    # DEBUG ACCURACY
    svm_pred = best_svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print("svm_accuracy:", svm_accuracy)

    return best_svm

def evaluate_model():
    pass

# Main
if __name__ == "__main__":
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")
    
    # Step 1. Preprocess Data
    print("Preprocessing data...")
    X = preprocess_data(X)
    
    # Note: fix a seed for reproducibility
    seed = 143 # Å

    # Step 2. Split processed data
    print("Splitting data...")
    model_dict = create_train_test_valid_sets(X, y, seed)

    X_train = model_dict["X"]["train"]
    y_train = model_dict["y"]["train"]
    X_test = model_dict['X']['test']
    y_test = model_dict['y']['test']

    # Step 3. Create 3 test models
    print("Creating 3 candidate models...")
    models = create_all_models(X_train, y_train, X_test, y_test, seed)
    
    # Step 4. Select model
    print("Choosing best model...")
    choose_best_model(models, X_test, y_test)

    # Step 5. Evaluate model
    print("Evaluating model...")
    
    # Step 6. Print out numbers like accuracy. Should be reproducible across runs
