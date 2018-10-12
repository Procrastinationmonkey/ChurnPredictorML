
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import keras
from keras.models import Sequential #initialize neural network
from keras.layers import Dense #adding layers


# Importing the dataset
data = pd.read_csv(r'C:\Users\Dell\Desktop\ChurnPrediction\data\Churndata.csv')

#independent variables
x = data.iloc[:, 3:13].values

#dependent column
y = data.iloc[:, 13].values

#categorical data encoding
labelencoder_X_1 = LabelEncoder()

x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])

labelencoder_X_2 = LabelEncoder()

x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])

x = onehotencoder.fit_transform(x).toarray()

x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising ANN
ann = Sequential()

# Adding the input layer and the first hidden layer 
#rectifier activation function for inputs
ann.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11)) 
#11 input nodes-independent variables 
#6 nodes in hidden layer=Average of nodes in input&output layer,small numbers close 
#to zero and weight initiazlized acc. to unifrom function
# Adding the second hidden layer

ann.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer #sigmoid activation function for output
ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer=algorithm to find optimal set of weights -stochastic gradient descent model selected
#loss function to optimize for sgd=logarithmic loss function(binary outcome)
#metrics=criterion to evaluate model and improve model performance each cycle

# Fitting the ANN to the Training set
ann.fit(X_train, y_train, batch_size = 5, nb_epoch = 50)
#batch size=no. of observations after which weights updated

prediction = ann.predict(X_test)
prediction = (prediction > 0.6)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)


