
# coding: utf-8

# In[ ]:

# Datenstruktur Library
import pandas as pd
# Machine Learning Library
from sklearn import linear_model
# Plotting Library
import matplotlib.pyplot as plt

# Test Daten einlesen
dataframeTest = pd.read_csv('test.txt', sep=';')
# Zuweisung von Spalten
x_values = dataframeTest[['Brain']]
y_values = dataframeTest[['Body']]

# Training des Modells
# Erzeugung Lineare Regression
body_reg = linear_model.LinearRegression()
# Übergabe der Daten für Training
body_reg.fit(x_values, y_values)

# Visualisierung der Daten
# Punkte des Datensatzes
plt.scatter(x_values, y_values)
# Vorhersage Graph
plt.plot(x_values, body_reg.predict(x_values))
plt.show()