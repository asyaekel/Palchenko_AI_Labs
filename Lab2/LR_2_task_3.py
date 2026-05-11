import pandas as pd
import numpy as np
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Завантажила датасет безпосередньо з репозиторію UCI
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# ЕТАП 1: ВІЗУАЛІЗАЦІЯ ДАНИХ (Ті самі 3 типи графіків)

# 1. Діаграма розмаху (Box and Whiskers)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.suptitle("Діаграма розмаху для кожної ознаки")
pyplot.show()

# 2. Гістограми розподілу
dataset.hist()
pyplot.suptitle("Гістограми розподілу ознак")
pyplot.show()

# 3. Матриця діаграм розсіювання
scatter_matrix(dataset)
pyplot.suptitle("Матриця діаграм розсіювання")
pyplot.show()

# ЕТАП 2: ПІДГОТОВКА ДАНИХ

# Розділила дані на ознаки та цільову змінну
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# Сформувала навчальну та контрольну вибірки (80/20)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# ЕТАП 3: ПОРІВНЯННЯ МОДЕЛЕЙ

# Підготувала список алгоритмів (виправила параметри для LogisticRegression)
models = [
    ('LR', LogisticRegression(solver='lbfgs', max_iter=1000)),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]

# Оцінила кожну модель за допомогою крос-валідації
results = []
names_list = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names_list.append(name)
    print(f'{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})')

# Побудувала графік порівняння точності алгоритмів
pyplot.boxplot(results, labels=names_list)
pyplot.title('Порівняння точності алгоритмів')
pyplot.show()

# ЕТАП 4: ФІНАЛЬНЕ ПЕРЕДБАЧЕННЯ

# Вибрала SVM як одну з найбільш стабільних моделей для цього датасету
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Вивела звіти про якість класифікації
print("\nМетрики на контрольній вибірці:")
print(f"Акуратність: {accuracy_score(Y_validation, predictions):.4f}")
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# ЕТАП 5: ПРОГНОЗ ДЛЯ НОВОЇ КВІТКИ (Крок 8 за завданням)
# Створила масив із новими вимірюваннями
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model.predict(X_new)

print(f"\nПараметри нової квітки: {X_new}")
print(f"Спрогнозований клас: {prediction[0]}")