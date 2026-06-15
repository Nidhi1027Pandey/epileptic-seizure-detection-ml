import pandas as pd

# load dataset
data = pd.read_csv("C:/Users/hprai/Downloads/archive/Epileptic Seizure Recognition.csv")

# print(data.head()) # Prints first 5 rows
print(data.shape) # tells the number of rows and columns the data has

# integer-location based indexing.selects rows and columns by position (numbers) instead of names.
data = data.iloc[0:, 1:] #start from 0th row till end, and 1st column till end--> take everything except first column(X12.V1.764..etc.)
# print(data.head())
print(data.shape)
# Separating features(inputs) and labels(outputs) by droping y column from X variable and having Y as just y column
X = data.drop(columns=['y'])
y = data['y']
# X = EEG signal features
# y = seizure / non-seizure 

# assigns zero to non seizure labels(2,3,4,5) and 1 to seizure label(1) so that it becomes binary--> seizure/nonseizure
#lambda is the function that is short for an if else loop
#apply is the function that makes sure that the lambda function is applied on every value of y 
y = y.apply(lambda x: 1 if x == 1 else 0)

from sklearn.model_selection import train_test_split #used to split the dataset (both feature and label) into two parts- training and testing
# if model tests on the same data it trained on, there is chance of it memorizing the output
#  testing dataset is used to see i ftrained model genralises to new unseen data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, #splits- 20% data for testing and 80% will be used for training
    random_state=42 # to make sure everytime the splitting is not different. 42 is just a arbitrary value (commonly used seed value)
)
from sklearn.ensemble import RandomForestClassifier # machine-learning algorithm that works by building many decision trees and combining their predictions. Think of it like asking many experts (trees) to vote on the answer.
# eg-Tree 1 → seizure
# Tree 2 → non-seizure
# Tree 3 → seizure
# Tree 4 → seizure
# Final decision = seizure (majority vote).
# This makes the model: more accurate, less likely to overfit

model = RandomForestClassifier() # creating the classifier object
# Model exists but it has not learned anything yet. Just sets up algorithm with default parameters like- no of trees, depth of trees, splitting criteria

model.fit(X_train, y_train) # Learning step
# Now the classifier is trained
y_pred = model.predict(X_test) #predicts label(seizure/non-seizure or 1/0) for test dataset

from sklearn.metrics import accuracy_score, classification_report # accuracy score-measures overall correctness  Classification report- gives detailedperformance breakdown

accuracy = accuracy_score(y_test, y_pred) #y_test is actual labels  y_pred is labels predicted by model

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(name, "Accuracy:", acc)

import matplotlib.pyplot as plt

importances = model.feature_importances_

plt.plot(importances)
plt.title("Feature Importance")
plt.xlabel("EEG Feature Index")
plt.ylabel("Importance")
plt.show()


from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.title("ROC Curve (AUC = %.2f)" % roc_auc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print("Cross Validation Accuracy:", scores.mean())