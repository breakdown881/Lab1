import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pprint import pprint
import io
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('D:/Machine Learning/KNN-IrisDataSet-master/diabetes.csv')
df.head()

df.columns = ["pregnancies", "glucose", "blood_pressure", "skin_thickness","insulin","bmi","Diabetes_Pedigree_Function","age","outcome"]
df.head()
df.glucose.replace(0,np.nan,inplace = True)
df.insulin.replace(0,np.nan,inplace = True)
df.blood_pressure.replace(0,np.nan,inplace = True)
df.bmi.replace(0,np.nan,inplace = True)
df.skin_thickness.replace(0,np.nan,inplace = True)
df.age.replace(0,np.nan,inplace = True)
df.Diabetes_Pedigree_Function.replace(0,np.nan,inplace = True)
df = df.fillna(df.mean())
df.head()
df.describe()
from sklearn.preprocessing import scale
df['insulin'] = scale(df['insulin'])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import svm
y = df['outcome'].values
X = df.drop('outcome',axis =1).values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y)
import matplotlib.pyplot as plt
import pylab
import numpy as np
neighbors  = np.arange(1,10)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,y_train)
    train_accuracy[i] = knn.score(X_train,y_train)
    test_accuracy[i] = knn.score(X_test,y_test)

knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(X_train,y_train)
y_pred_1 = knn.predict(X_test)
knn.score(X_test,y_test)
accuracy = accuracy_score(y_test, y_pred_1)
recall = recall_score(y_test, y_pred_1, average="weighted")
precision = precision_score(y_test, y_pred_1, average="weighted")
print("============= KNN Results =============")
print("Accuracy    : ", accuracy)
print("Recall      : ", recall)
print("Precision   : ", precision)
svm = svm.SVC()
svm.fit(X_train, y_train)
y_pred_1 = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_1)
recall = recall_score(y_test, y_pred_1, average="weighted")
precision = precision_score(y_test, y_pred_1, average="weighted")
print("============= SVM Results =============")
print("Accuracy    : ", accuracy)
print("Recall      : ", recall)
print("Precision   : ", precision)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=70, # Number of trees
                                  min_samples_split = 30,
                                  bootstrap = True, 
                                  max_depth = 50, 
                                  min_samples_leaf = 25)
rf_model.fit(X_train,y_train)
result = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_1)
recall = recall_score(y_test, y_pred_1, average="weighted")
precision = precision_score(y_test, y_pred_1, average="weighted")
print("============= RandomForest Results =============")
print("Accuracy    : ", accuracy)
print("Recall      : ", recall)
print("Precision   : ", precision)
dct = DecisionTreeClassifier()
# Train Decision Tree Classifer
dct = dct.fit(X_train,y_train)
#Predict the response for test dataset
y_pred_3 = dct.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_3)
recall = recall_score(y_test, y_pred_3, average="weighted")
precision = precision_score(y_test, y_pred_3, average="weighted")
print("============= DecisionTree Results =============")
print("Accuracy    : ", accuracy)
print("Recall      : ", recall)
print("Precision   : ", precision)