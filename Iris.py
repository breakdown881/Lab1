import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

#Feature Selection Libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# ML Libraries (Random Forest, Naive Bayes, SVM)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
 
# Evaluation Metrics
from sklearn import metrics
df = pd.read_csv('Iris.csv', error_bad_lines=False)
df['Species'].unique()
df['Species'] = pd.factorize(df["Species"])[0] 
Target = 'Species'
df['Species'].unique()
df = df.drop(['Id'], axis=1)
Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X_fs = df[Features]
Y_fs = df[Target]
model = LogisticRegression(solver='lbfgs', multi_class='auto')

#Mark the Number of Features to be selected, Adjust this Number to enhance the Model Performance
rfe = RFE(model, 3) 
fit = rfe.fit(X_fs, Y_fs)
Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]    #Features to train
x2 = x[Target]      #Target Class to train
y1 = y[Features]    #Features to test
y2 = y[Target]      #Target Class to test
# Gaussian Naive Bayes
# Create Model with configuration
nb_model = GaussianNB() 

# Model Training
nb_model.fit(X=x1, y=x2)

# Prediction with Test Set
result= nb_model.predict(y[Features]) 
# Random Forest
# Create Model with configuration
rf_model = RandomForestClassifier(n_estimators=70, # Number of trees
                                  min_samples_split = 30,
                                  bootstrap = True, 
                                  max_depth = 50, 
                                  min_samples_leaf = 25)

# Model Training
rf_model.fit(X=x1,
             y=x2)

# Prediction
result = rf_model.predict(y[Features])
# Model Evaluation
ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("========== Random Forest Results ==========")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)
# Classification Report
# Instantiate the classification model and visualizer
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# visualizer = ClassificationReport(rf_model, classes=target_names)
# visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
# visualizer.score(y1, y2)       # Evaluate the model on the test data

print('================= Classification Report =================')
print('')
print(classification_report(y2, result, target_names=target_names))
# Support Vector Machine
# Create Model with configuration
svm_model = SVC(kernel='linear')

# Model Training
svm_model.fit(X=x1, y=x2)  

# Prediction
result = svm_model.predict(y[Features]) 
# Model Evaluation
ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)

print("============= SVM Results =============")
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)
# Classification Report
# Instantiate the classification model and visualizer
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# visualizer = ClassificationReport(svm_model, classes=target_names)
# visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
# visualizer.score(y1, y2)       # Evaluate the model on the Test Set

print('================= Classification Report =================')
print('')
print(classification_report(y2, result, target_names=target_names))
