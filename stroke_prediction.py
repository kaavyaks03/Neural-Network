import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns #Library files are imported 

tf.__version__
dataset = pd.read_csv('C:/Users/Admin/Desktop/ANN/stroke_final.csv')  #data set is read in csv file
dataset.head()

X = dataset.iloc[:, 2:-1].values   #Data set values are seperated 
y = dataset.iloc[:, -1].values

print(X.shape)
print(X)  

print(y)
print("\n")

from sklearn.preprocessing import LabelEncoder   #the categorial values are encoded as numerical values
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
X[:, -1] = le.fit_transform(X[:, -1])
print(X)
print("\n")


from sklearn.compose import ColumnTransformer   #the numerical values are fixed between a given range for  further processing
from sklearn.preprocessing import OneHotEncoder   #this is done using ONEHOTENCODER
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X.shape)
print("\n")
print(X)

from sklearn.model_selection import train_test_split   #data are split into training ans testing part of datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential() #implementing of neural network

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  #input layer-1 and hidden layer-1 is implemented
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))  #hidden layere-2 is implemented
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #output layer-1 is implemented

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #finding the accuracy for all the datas

ann.fit(X_train, y_train, batch_size = 25, epochs = 30)

print("\n")
print(ann.predict(sc.transform([[1, 55, 1, 0, 85, 20, 2]])) < 0.5) #training the neural network

print("\n")
y_pred_test = ann.predict(X_test)
y_pred_test = (y_pred_test < 0.5)
print(np.concatenate((y_pred_test.reshape(len(y_pred_test),1), y_test.reshape(len(y_test),1)),1)) #predicting the test dataset result

print("\n")
y_pred_train = ann.predict(X_train)
y_pred_train = (y_pred_train < 0.5)
print(np.concatenate((y_pred_train.reshape(len(y_pred_train),1), y_train.reshape(len(y_train),1)),1))#predicting the train dataset result

from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
print("\n")
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)
sns.heatmap(cm_test, square=True , annot=True) #makeing confusion matrix - test dataset

print("\n")
cm_train = confusion_matrix(y_train, y_pred_train)
print(cm_train)
sns.heatmap(cm_train, square=True , annot=True) #making confusiin matrix - train dataset