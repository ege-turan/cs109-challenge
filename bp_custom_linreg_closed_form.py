# Ege Turan
# mathematics learned from Probabilistic Machine Learning: An Introduction by Kevin Patrick Murphy. MIT Press, March 2022
# greatly inspired by Jay Pradip Shah's implementation here: https://www.kaggle.com/jaypradipshah/code

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
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test

# Custom class for training and testing a linear regression model
class linearRegression():

  def __init__(self):
    # No instance Variables
    pass

  def predict_and_loss(self,X,y,W):
      y_pred = np.dot(W, X)
      loss = np.mean((y - y_pred)**2)  # Calculate mean squared error (loss function)
      return loss, y_pred              # the prediction and how off the prediction was

  def update_weights(self,X,y_pred,y_true,W,learning_rate,index):
    x_with_bias = np.append(1,X[index])
    for i in range(1 + X.shape[1]):
      W[i] -= (learning_rate * (y_pred-y_true[index])*x_with_bias[i]) 
    return W
  
  def train_closed_form(self, X, y):
      W_optimal = np.linalg.inv(X.T @ X) @ X.T @ y
      return W_optimal

  def train(self, X, y, training_steps=1000, learning_rate=0.001, random_state=0):

    num_rows = X.shape[0]
    num_cols = X.shape[1]
    W = np.random.randn(1,num_cols) / np.sqrt(num_rows) # start with random weights

    # Calculating the losses, summing into costs. Using it to build a train_loss database
    train_loss = []
    num_trainings = []
    train_indices = [i for i in range(X.shape[0])]
    for j in range(training_steps):
      cost=0
      np.random.seed(random_state)
      np.random.shuffle(train_indices)
      for i in train_indices:
        loss, y_pred = self.predict_and_loss(X[i],y[i],W[0])
        cost += loss
        W[0] = self.update_weights(X,y_pred,y,W[0],learning_rate,i)
      train_loss.append(cost)
      num_trainings.append(j)
    return W[0], train_loss, num_trainings

  def test(self, X_test, y_test, W_trained):

    test_pred = []
    test_loss = []
    for i in range(X_test.shape[0]): #indices of tests
        loss, y_test_pred = self.predict_and_loss(X_test[i], W_trained, y_test[i])
        test_pred.append(y_test_pred)
        test_loss.append(loss)
    return test_pred, test_loss
    

  def plot_loss(self, loss, training_steps):
    plt.plot(training_steps, loss)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Loss')
    plt.title('Plot Loss')
    plt.show()

# helper for test_print
  def test_calc(self, X_test, y_true, W_trained):
    y_pred = np.dot(X_test, W_trained)
    mae = np.mean(abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    ssr = np.sum((y_true - y_pred)**2)  # Sum of squared residuals
    sst = np.sum((y_true - np.mean(y_true))**2)  # Total sum of squares
    r_squared = 1 - (ssr / sst)
    return test_pred, mae, mse, r_squared

  def test_print(self, X_test, y_test, W_trained):
    test_pred, mean_error, test_loss, r_squared = self.test_calc(X_test, y_test, W_trained)
    print(f"Mean Error (ME): {mean_error}")
    print(f"Mean Squared Error (MSE): {test_loss}")
    print(f"R^2 Score: {r_squared}")
    return test_pred, mean_error, test_loss, r_squared

# Splitting the dataset
X_train, y_train, X_test, y_test = split_data(X,y,test_size=0.2, random_state=42)

# Creating a "regressor" object of the class LinearRegression
regressor = linearRegression()

# Training (closed form)
W_trained = regressor.train_closed_form(X_train, y_train)

# Testing on the Test Dataset
test_pred, test_loss = regressor.test(X_test, y_test, W_trained)

# Testing on the Test Dataset
test_pred, mean_error, test_loss, r_squared = regressor.test_print(X_test, y_test, W_trained)

# Printing the Weights
print(W_trained)