import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

# Завантажила дані для пошуку
data = np.loadtxt('data_random_forests.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Визначила сітку параметрів для перебору
param_grid = [
    {'n_estimators': [25, 50, 100], 'max_depth': [2, 4, 8]},
    {'max_depth': [10, 15], 'n_estimators': [100, 200]}
]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print(f"Пошук найкращих параметрів для {metric}:")
    # Використала GridSearchCV для перебору комбінацій
    gs = GridSearchCV(ExtraTreesClassifier(random_state=0), param_grid, cv=5, scoring=metric)
    gs.fit(X_train, y_train)
    
    print(f"Найкращі параметри: {gs.best_params_}")
    y_pred = gs.predict(X_test)
    print(classification_report(y_test, y_pred))