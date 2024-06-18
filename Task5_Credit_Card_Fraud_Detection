import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load and preprocess the dataset
file_path = '/content/creditcard.csv'  # Adjust the path as necessary
file = pd.read_csv(file_path)

# Display first 10 rows of the dataframe
print(file.head(10))

# Display summary statistics of the dataframe
print(file.describe())

# Check for missing values
print(file.isnull().sum())

# Count the number of instances for each class
print(file['Class'].value_counts())

# Step 3: Separate normal and fraud transactions
normal = file[file['Class'] == 0]
fraud = file[file['Class'] == 1]

# Display shapes of normal and fraud transactions
print(normal.shape)
print(fraud.shape)

# Display descriptive statistics of transaction amounts for normal and fraud
print(normal['Amount'].describe())
print(fraud['Amount'].describe())

# Step 4: Balance the classes (if necessary)
# In this case, assuming the dataset is already balanced or we'll sample to balance classes
normal_sample = normal.sample(n=len(fraud), random_state=2)  # Ensure reproducibility with random_state

# Concatenate normal_sample and fraud to create a new balanced dataframe
new_file = pd.concat([normal_sample, fraud], axis=0)

# Display first 10 rows of new_file and count instances of each class
print(new_file.head(10))
print(new_file['Class'].value_counts())

# Step 5: Prepare data for modeling
X = new_file.drop(columns='Class', axis=1)
Y = new_file['Class']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Step 6: Initialize and train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter for convergence
model.fit(X_train, Y_train)

# Step 7: Evaluate model performance
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) * 100
print(f"Training Data Accuracy: {training_data_accuracy:.2f}%")

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) * 100
print(f"Test Data Accuracy: {test_data_accuracy:.2f}%")

# Display confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))

print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))
