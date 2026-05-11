import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

def visualize_classifier(classifier, X, y, title=''):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, 0.01), np.arange(min_y, max_y, 0.01))
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()]).reshape(x_vals.shape)
    plt.figure()
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.title(title)

parser = argparse.ArgumentParser(description='Ансамблевий класифікатор')
parser.add_argument('--classifier-type', dest='type', required=True, choices=['rf', 'erf'])
args = parser.parse_args()

# Завантажила дані
data = np.loadtxt('data_random_forests.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# ГРАФІК 1: Вхідні дані з різними маркерами
plt.figure()
markers = ['s', 'o', '^']
for i, marker in enumerate(markers):
    plt.scatter(X[y==i][:, 0], X[y==i][:, 1], marker=marker, facecolors='none', edgecolors='black', s=75)
plt.title("Вхідні дані")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
classifier = RandomForestClassifier(**params) if args.type == 'rf' else ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

# ГРАФІК 2: Межі на навчальних даних
visualize_classifier(classifier, X_train, y_train, 'Навчальна вибірка')

# ГРАФІК 3: Межі на тестових даних
visualize_classifier(classifier, X_test, y_test, 'Тестова вибірка')

print(classification_report(y_test, classifier.predict(X_test)))
plt.show()