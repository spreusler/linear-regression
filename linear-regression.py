
# coding: utf-8

# In[ ]:

# Datenstruktur Library
import pandas as pd
# Machine Learning Library
from sklearn import linear_model
# Plotting Library
import matplotlib.pyplot as plt

# Train Daten einlesen
dataframeTrain = pd.read_fwf('train.txt')
# Zuweisung von Spalten
x_valuesTrain = dataframeTrain[['Brain']]
y_valuesTrain = dataframeTrain[['Body']]

# Test Daten einlesen
dataframeTest = pd.read_csv('test.txt', sep=';')
# Zuweisung von Spalten
x_valuesTest = dataframeTest[['Brain']]
y_valuesTest = dataframeTest[['Body']]

# Training des Modells

# Erzeugung Lineare Regression
body_regTrain = linear_model.LinearRegression()
# Übergabe der Daten für Training
body_regTrain.fit(x_valuesTrain, y_valuesTrain)

# Erzeugung Lineare Regression
body_regTest = linear_model.LinearRegression()
# Übergabe der Daten für Training
body_regTest.fit(x_valuesTest, y_valuesTest)

# Visualisierung der Daten

# Punkte des Train Datensatzes
plt.scatter(x_valuesTrain, y_valuesTrain)
# Punkte des Test Datensatzes
plt.scatter(x_valuesTest, y_valuesTest, color='red')

# Train Graph
plt.plot(x_valuesTrain, body_regTrain.predict(x_valuesTrain))
# Test Graph
plt.plot(x_valuesTest, body_regTest.predict(x_valuesTest), color='red')
plt.show()