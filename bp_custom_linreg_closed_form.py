# Ege Turan

# Acknowledgements and Citations:
# Mathematics learned from Probabilistic Machine Learning: An Introduction by Kevin Patrick Murphy. MIT Press, March 2022
# Closed-form training inspired by Chris Piech's comment about why sk-learn is fast
# Custom regressing model inspired by Jay Pradip Shah's implementation here: https://www.kaggle.com/jaypradipshah/code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from pwdb_haemod_params.csv
dataset = pd.read_csv(r'BP\pwdb_haemod_params.csv')

# Select relevant columns for analysis
columns_of_interest = [' age [years]', ' HR [bpm]', ' PWV_fa [m/s]', ' Tr [ms]', ' SBP_a [mmHg]']
# columns_of_interest = [' age [years]', ' HR [bpm]', ' PWV_fa [m/s]', ' Tr [ms]', 
#                       ' SBP_b [mmHg]', ' PP_a [mmHg]', ' PP_b [mmHg]', ' PWV_a [m/s]', ' MAP_a [mmHg]',
#                       ' SBP_a [mmHg]']

# Subset the data with relevant columns
selected_data = dataset[columns_of_interest]

# Drop rows with missing values
selected_data = selected_data.dropna()

# Split the data into features (X) and target (y)
X = np.asarray(dataset[[' age [years]', ' HR [bpm]', ' PWV_fa [m/s]', ' Tr [ms]']].values.tolist())
#X = np.asarray(dataset[[' age [years]', ' HR [bpm]', ' PWV_fa [m/s]', ' Tr [ms]', 
#                       ' SBP_b [mmHg]', ' PP_a [mmHg]', ' PP_b [mmHg]', ' PWV_a [m/s]', ' MAP_a [mmHg]',
#                       ' SBP_a [mmHg]']].values.tolist()) 
y = np.asarray(dataset[' SBP_a [mmHg]'].values.tolist())

# The helper method "split_data" splits the given dataset into a training set and a test set
# This is similar to the method "train_test_split" from "sklearn.model_selection"
def split_data(X,y,test_size=0.2,random_state=0):
    
    np.random.seed(random_state)                  # seed for reproducible results
    indices = np.random.permutation(len(X))       # shuffling the indices to be split

    data_test_size = int( X.shape[0] * test_size )  # Get the test size (according to proportion)

    train_indices = indices[data_test_size:]
    test_indices = indices[:data_test_size]

    X_train = X[train_indices]
    X_test = X[test_indices]
    
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test

# Custom class for training and testing a linear regression model
class linearRegression():

  def __init__(self):
    # No instance Variables
    pass

  def train_closed_form(self, X, y):
    W_optimal = np.linalg.inv(X.T @ X) @ X.T @ y
    return W_optimal

  def update_weights(self,X,pred,true,W,learning_rate,index):
    x_with_bias = np.append(1,X[index])

    for i in range(1 + X.shape[1]):
      gradient = (pred-true[index])*x_with_bias[i]
      W[i] -= (learning_rate * gradient) 

    return W
  
  def shuffle_train_indices(self, train_indices, random_state=0):
      np.random.seed(random_state)
      np.random.shuffle(train_indices)
      return train_indices

  def train(self, X, y, training_steps=1000, learning_rate=0.001, random_state=0):

    W = np.random.randn(1,X.shape[0]) / np.sqrt(1 + X.shape[1]) # start with random weights

    # Calculating the losses, summing into costs. Using it to build a train_loss database
    train_loss = []
    train_indices = [i for i in range(X.shape[0])]
    num_trainings = [s for s in range(training_steps)]

    for j in range(training_steps):

      cost = 0

      shuffled_train_indices = self.shuffle_train_indices(train_indices, random_state)

      for i in shuffled_train_indices:
        
        pred = np.dot( W[0], X[i] )
        diff = y[i] - pred
        loss = np.mean( abs(diff) ** 2 )  # Calculate mean squared error (loss function)

        W[0] = self.update_weights(X,pred,y,W[0],learning_rate,i)
        cost += loss
        
      train_loss.append(cost)

    return W[0], train_loss, num_trainings

  def test(self, X_test, y_test, W_trained):

    pred = []
    loss = []

    for i in range(X_test.shape[0]): # indices of tests
        
        pred = np.dot( W_trained, X_test[i] )
        diff = y_test[i] - pred
        loss = np.mean( abs(diff) ** 2 )  # Calculate mean squared error (loss function)

        pred.append(pred)
        loss.append(loss)

    return pred, loss
    

  def training_plot(self, loss, training_steps):
    plt.plot(training_steps, loss)
    plt.title('Training Plot')
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Losses')
    plt.show()

# helper for test_print
  def test_calc(self, X_test, true, W_trained):
    pred = np.dot(X_test, W_trained)
    mae = np.mean(abs(true - pred))
    mse = np.mean((true - pred)**2)
    ssr = np.sum((true - pred)**2)  # Sum of squared residuals
    sst = np.sum((true - np.mean(true))**2)  # Total sum of squares
    r_squared = 1 - (ssr / sst)
    return test_pred, mae, mse, r_squared

  def test_print(self, X_test, y_test, W_trained):
    test_pred, mean_error, test_loss, r_squared = self.test_calc(X_test, y_test, W_trained)
    print(f"Mean Error (ME): {mean_error}")
    print(f"Mean Squared Error (MSE): {test_loss}")
    print(f"R^2 Score: {r_squared}")
    return test_pred, mean_error, test_loss, r_squared

# Splitting the dataset
X_train, y_train, X_test, y_test = split_data(X,y,test_size=0.2, random_state=0)

# Creating an object of the class LinearRegression
model = linearRegression()

# Training (closed form)
W_trained = model.train_closed_form(X_train, y_train)

# Testing on the Test Dataset
test_pred, test_loss = model.test(X_test, y_test, W_trained)

# Testing on the Test Dataset
test_pred, mean_error, test_loss, r_squared = model.test_print(X_test, y_test, W_trained)

# Printing the Weights
print(W_trained)
