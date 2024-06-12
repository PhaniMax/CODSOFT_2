import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
iris = pd.read_csv("IRIS.csv")

print(iris.head())
print()

print(iris.describe())
print()

print("Target Labels", iris["species"].unique())
print()

import plotly.express as px
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()
print()

x = iris.drop("species", axis=1)
y = iris["species"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

x_new = pd.DataFrame([[5, 2.9, 1, 0.2]], columns=x.columns)
prediction = knn.predict(x_new)
print()
print("Prediction: {}".format(prediction))
