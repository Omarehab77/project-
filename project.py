import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
df = pd.read_csv("C:/Users/omero/Downloads/طك.csv")
print(df.shape)
print(df.head(0))
print(df.nunique())
print("----------------------------")
print(df.dtypes)
print(df.Model.unique())
print(df.shape)
print(df.dropna().shape)
print(df.isnull().sum())
print(df.describe())
corr_matrix = df.corr()
sns.heatmap(corr_matrix)
print(df.isna().sum())
plt.show()
plt.hist(df['Postal Code'])
plt.show()
sns.scatterplot(x='Postal Code', y='Electric Range', data=df)
plt.show()
corr = df['Postal Code'].corr(df['Electric Range'])
print("Correlation: ", corr)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
