import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Завантажила діабет-датасет
diabetes = datasets.load_diabetes()
X, y = diabetes.data, diabetes.target

# Розбила вибірку навпіл
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Побудувала регресію
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

# Вивела результати
print("Коефіцієнти:", regr.coef_)
print("R2 score:", round(r2_score(y_test, y_pred), 2))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))

# Побудувала графік
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.xlabel('Виміряно')
plt.ylabel('Передбачено')
plt.show()