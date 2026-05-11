import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Дані з попереднього завдання
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        train_errors.append(mean_squared_error(y_train[:m], model.predict(X_train[:m])))
        val_errors.append(mean_squared_error(y_val, model.predict(X_val)))
    plt.plot(np.sqrt(train_errors), "r-+", label="train")
    plt.plot(np.sqrt(val_errors), "b-", label="val")
    plt.ylim(0, 3)
    plt.legend()

# Крива для лінійної моделі
plt.figure()
plot_learning_curves(LinearRegression(), X, y)
plt.title("Linear Learning Curves")

# Крива для полінома 10 ступеня
plt.figure()
poly_10 = Pipeline([("poly", PolynomialFeatures(degree=10)), ("lin", LinearRegression())])
plot_learning_curves(poly_10, X, y)
plt.title("Polynomial (deg 10) Learning Curves")

plt.show()