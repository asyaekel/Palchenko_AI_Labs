import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from io import BytesIO

# Підготувала дані
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Навчила Ridge класифікатор (виправила імена змінних)
clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Розрахувала розширені метрики
print('Accuracy:', np.round(metrics.accuracy_score(y_test, y_pred), 4))
print('Cohen Kappa:', np.round(metrics.cohen_kappa_score(y_test, y_pred), 4))
print('Matthews Corr:', np.round(metrics.matthews_corrcoef(y_test, y_pred), 4))

# Побудувала матрицю помилок
sns.set()
mat = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig("Confusion.jpg")
plt.show()