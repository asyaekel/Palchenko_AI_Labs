import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Завантажила датасет цін на житло (замість Boston)
data = fetch_california_housing()
X, y = shuffle(data.data, data.target, random_state=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Побудувала AdaBoost регресор
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Отримала та нормалізувала важливість ознак
importances = regressor.feature_importances_
importances = 100.0 * (importances / max(importances))
sorted_idx = np.flipud(np.argsort(importances))

# Побудувала діаграму
pos = np.arange(sorted_idx.shape[0]) + 0.5
plt.figure()
plt.bar(pos, importances[sorted_idx], align='center')
plt.xticks(pos, np.array(data.feature_names)[sorted_idx], rotation=45)
plt.ylabel('Відносна важливість')
plt.title('Важливість ознак в AdaBoost')
plt.tight_layout()
plt.show()