import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Завантажила дані з файлу
input_file = 'data_singlevar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбила на навчальну (80%) та тестову вибірки
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Створила та навчила лінійну модель
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Зробила прогноз
y_pred = regressor.predict(X_test)

# Візуалізувала результат
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='black', linewidth=3)
plt.title('Регресія однієї змінної')
plt.show()

# Вивела метрики якості
print("Метрики лінійної регресії:")
print("MAE =", round(sm.mean_absolute_error(y_test, y_pred), 2))
print("MSE =", round(sm.mean_squared_error(y_test, y_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

# Зберегла модель у файл
with open('model.pkl', 'wb') as f:
    pickle.dump(regressor, f)