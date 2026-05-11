import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Згенерувала дані за 3-м варіантом
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Поліноміальна регресія (ступінь 2)
poly_feat = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_feat.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Візуалізація
plt.scatter(X, y, color='gray')
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
plt.plot(X_range, lin_reg.predict(X_range), 'r-', label='Linear')
plt.plot(X_range, poly_reg.predict(poly_feat.transform(X_range)), 'b-', label='Polynomial')
plt.legend()
plt.show()

print("Варіант 3: Коефіцієнти полінома", poly_reg.coef_)