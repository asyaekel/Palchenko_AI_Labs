import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()]).reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.title(title)

# Завантажила дисбалансні дані
data = np.loadtxt('data_imbalance.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# ГРАФІК 1: Візуалізація вхідного дисбалансу
plt.figure()
plt.scatter(X[y==0][:,0], X[y==0][:,1], marker='x', color='black', label='Class 0')
plt.scatter(X[y==1][:,0], X[y==1][:,1], marker='o', facecolors='none', edgecolors='red', label='Class 1')
plt.title("Вхідний дисбаланс класів")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# ГРАФІК 2: Без балансування
clf_no_balance = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
clf_no_balance.fit(X_train, y_train)
visualize_classifier(clf_no_balance, X_test, y_test, 'Без балансування ваг')
print("Звіт без балансування:\n", classification_report(y_test, clf_no_balance.predict(X_test), zero_division=0))

# ГРАФІК 3: З балансуванням (class_weight='balanced')
clf_balanced = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0, class_weight='balanced')
clf_balanced.fit(X_train, y_train)
visualize_classifier(clf_balanced, X_test, y_test, 'З балансуванням класів')
print("Звіт з балансуванням:\n", classification_report(y_test, clf_balanced.predict(X_test)))

plt.show()