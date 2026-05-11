import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Підготувала багатовимірні дані
data = np.loadtxt('data_multivar_regr.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Лінійна модель
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train, y_train)

# Поліноміальна модель (ступінь 10)
poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
poly_reg = linear_model.LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Перевірка на контрольній точці
point = [[7.75, 6.35, 5.56]]
print("Прогноз (Linear):", lin_reg.predict(point))
print("Прогноз (Polynomial):", poly_reg.predict(poly.transform(point)))