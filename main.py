import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report as crpt
from sklearn.metrics import confusion_matrix as cmat

cancer = load_breast_cancer()
#print(cancer.keys())

#print(cancer['data'].shape) # gives data size of 569 by 30

X, y = cancer['data'], cancer['target']
#print(X)

# We have our data. We now need to split into sections for training
# validating and test (or just train and test)

X_train, X_test, y_train, y_test = train_test_split(X, y)
#print(X_train)

#Apparently sklearn can have problems with non-normalised data 
# therefore we are required to do this
scaler = StandardScaler()
scaler.fit(X_train)
#X_train_n, X_test_n, y_train_n, y_test_n = ssc(copy=True, with_mean=True, with_std=True)

#Apply transformations
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#create the model - using number of features as number of vertices in each layer
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30))

#train model
mlp.fit(X_train, y_train)

#Use model
predictions = mlp.predict(X_test)

print(cmat(y_test,predictions))
print(crpt(y_test, predictions))