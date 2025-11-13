import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Diabetes.csv")

X = data[['Glucose']]
y = data['DiabetesPedigreeFunction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

lr = LinearRegression()
model = lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.scatter(X_test, sorted(y_test))
plt.plot(X_test, sorted(y_pred), color='red')
plt.show()

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(model, f)