import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Завантажила файл для 3-го варіанту
data = np.loadtxt('data_regr_3.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розподілила дані
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Навчила регресор
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Передбачила значення
y_pred = model.predict(X_test)

# Побудувала графік
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print("R2 score для 3-го варіанту:", round(sm.r2_score(y_test, y_pred), 2))