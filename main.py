# INF264 Project 2
# Phillip Lei and Ryan Huynh

# TODO:
# - Look at misclassified examples
# - Create model visualizations
# - Create automated test pipeline

# Import libraries
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as skimg

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score

# Functions
def create_train_test_valid_sets(X, y, seed):
    """
    Parameters:
     - X and y are the input dataset
     - seed: an integer for the randomization seed to ensure reproducibility

    Return value: a dictionary containing train, test, and validation sets for X and y
    """
    print("Splitting data into training, test, and validation sets...")

    train_set_size = 0.8
    test_size = 1 - train_set_size

    print(f"training set size: {(train_set_size * 100):.1f}%")
    print(f"test set size: {(test_size * 100):.1f}%")

    # Create training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Store splits into a dictionary structure
    model_dict = {
        "X": {
            "train": X_train,
            "test": X_test
        },
        "y": {
            "train": y_train,
            "test": y_test
        }
    }

    print("Data successfully split into train, test, and validation sets")
    return model_dict

"""
This function is supposed to score each candidate model through cross validation 
and return the one with the best score.

Return: model with the best score
"""
def choose_best_model(candidates, X_test, y_test):
    """
    Selects the best model out of the three
    """
    print("Choosing best model...")

    best_model = None
    best_score = 0
    
    # Iterate through candidate models 
    for name in candidates:
        # 10 is a common cross validation fold value
        model = candidates[name]
        mean_score = cross_val_score(model, X_test, y_test, cv=10).mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = model
            
    print("Best model: ", model)
    print("Score of model: ", best_score)
    
    return best_model

"""
Visualizes an image from the X array

Parameters:
    - X_entry is an entry from the array X
"""
def visualize_image(X_entry):
    # Helper function to view an image
    plt.imshow(X_entry.reshape(20,20), vmin=0, vmax=255, cmap="gray")
    plt.show()

"""
Process data by setting image pixels to binary values and scaling images down.
Intended to help computation run time.

Output: array of processed image lists
"""
def preprocess_data(X):
    print("Preprocessing data...")
    start_time = time.time()

    processed_X = []
    
    # For every image, set the image pixels to binary values
    # and downsize the image in order to improve computation
    for image in X:
        image = normalize_image_rgb(image)
        image = resize_image(image)
        processed_X.append(image)

    total_time = time.time() - start_time
    print(f'Finished preprocessing in {total_time:.2f} seconds')
    return processed_X

"""
Shrinking an image due to running time constraints. Smaller images (np arrays) 
should help our training run faster.

Output: resized np array representing an image
"""
def resize_image(image):
    # Turn image into np array if it is not
    if isinstance(image, np.ndarray):
        image = np.array(image)
        
    image = image.reshape(20, 20)
    # Max pooling function to reduce image by a factor of 2, 
    # turning the original image into 1/4th of the size
    resized_image = skimg.block_reduce(image, (2, 2), np.max) 
    
    return resized_image.reshape(100)
    
"""
Normalize pixel values of the image to 0 or 1.

Output: image with each value being binary
"""
def normalize_image_rgb(image):
    # Assign a pixel a 1 or 0 based off the middle threshold:
    # between 0 and 255
    border = 127
    for i in range(len(image)):
        if image[i] > border:
            image[i] = 1
        else:
            image[i] = 0
            
    return image    

"""
Responsible for creating the three candidate models: 
nn, decision tree, support vector machine.

Output: returns data structure of the three models
"""
def create_all_models(X_train, y_train, X_test, y_test, seed, tuning):
    print("Creating 3 candidate models...")

    # 1: NEURAL NETWORK
    nn = create_nn(X_train, y_train, X_test, y_test, seed, tuning)
    # 2: DECISION TREE
    dtree = create_dtree(X_train, y_train, X_test, y_test, seed, tuning)
    # 3: SUPPORT VECTOR MACHINE
    svm = create_svm(X_train, y_train, X_test, y_test, seed, tuning)
    
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

    neural_network = MLPClassifier(max_iter= 500, random_state = seed, activation='logistic',
                                   learning_rate='constant', solver='adam')

    # Create a tuning list for grid search to find the best of them
    if tuning:
        print("NN hyperparameter tuning...")
        parameter_grid = {
            "activation": ['logistic', 'relu'],
            "solver": ['sgd', 'adam'],
            "learning_rate": ['constant', 'adaptive']
        }
        
        # Grid search finds the best of the hyperparameters
        # and fits/trains models to find the best performance
        grid_search = GridSearchCV(neural_network, parameter_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best Neural Network Parameters: " + str(best_params))
        best_nn = grid_search.best_estimator_
        best_nn = neural_network.fit(X_train, y_train)
    else:
        best_nn = neural_network.fit(X_train, y_train)
    
    total_time = time.time() - start_time
    print("Neural network create")
    print(f"Finished in {total_time:.2f} seconds")

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

    dtree = DecisionTreeClassifier(random_state=seed, criterion='entropy', max_depth=17,
                                   splitter='random')
    
    # Create a tuning list for grid search to find the best of them
    if tuning: 
        print("Decision Tree hyperparameter tuning...")
        parameter_grid = {
            "max_depth": [17, 20],
            "criterion": ["gini", "entropy"],
            "splitter": ["best", "random"]
        }
        
        # Grid search finds the best of the hyperparameters
        # and fits/trains models to find the best performance
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
    print(f"Finished in {total_time:.2f} seconds")

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

    svm = SVC(random_state=seed, C=10, gamma='scale', kernel='rbf')
    
    # Create a tuning list for grid search to find the best of them
    if tuning:
        print("SVM hyperparameter tuning...")
        parameter_grid = {
            'C': [1, 10],  
            'gamma': ["auto", "scale"], 
            'kernel': ['rbf', 'poly']
        }
    
        # Grid search finds the best of the hyperparameters
        # and fits/trains models to find the best performance
        grid_search = GridSearchCV(svm, parameter_grid, cv=3)
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        print("Best SVM Parameters: " + str(best_params))
        best_svm = grid_search.best_estimator_
    else:
        best_svm = svm.fit(X_train, y_train)

    total_time = time.time() - start_time
    print("SVM created")
    print(f"Finished in {total_time:.2f} seconds")

    # DEBUG ACCURACY
    svm_pred = best_svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    print("svm_accuracy:", svm_accuracy)

    return best_svm

"""
Add explanation
"""
def evaluate_model(best_model, X_test, y_test):
    print("Evaluating model...")
    predictions = best_model.predict(X_test)

    return accuracy_score(y_test, predictions)

"""
Visualizes the decision tree up to a max depth
"""
def visualize_decision_tree(tree):
    plt.figure(figsize=(12,8))
    plot_tree(tree, filled=True, label='all', impurity=False, max_depth=2, fontsize=7)
    plt.show()

# Main
if __name__ == "__main__":
    X = np.load("emnist_hex_images.npy")
    y = np.load("emnist_hex_labels.npy")
    
    # Step 1. Preprocess Data
    X = preprocess_data(X)
    
    # Note: fix an arbitrary seed for reproducibility
    seed = 143 

    # Step 2. Split processed data
    model_dict = create_train_test_valid_sets(X, y, seed)

    X_train = model_dict["X"]["train"]
    y_train = model_dict["y"]["train"]
    X_test = model_dict['X']['test']
    y_test = model_dict['y']['test']

    # Step 3. Create 3 test models
    tuning = False
    models = create_all_models(X_train, y_train, X_test, y_test, seed, tuning)
    
    # Step 4. Select model
    best_model = choose_best_model(models, X_test, y_test)

    # Step 5. Evaluate model
    accuracy = evaluate_model(best_model, X_test, y_test)
    
    # Step 6. Print out numbers like accuracy. Also plots and visualizations
    print("Accuracy of evalution: ", accuracy)

    # Visualize decision tree
    visualize_decision_tree(models["Decision Tree"])

    # Visualize confusion matrix for best model
    confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
    print(confusion_matrix)


