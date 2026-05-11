import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Завантажила дані про трафік
data = []
with open('traffic_data.txt', 'r') as f:
    for line in f.readlines():
        data.append(line.strip().split(','))
data = np.array(data)

# Створила та навчила кодувальники для текстових даних
label_encoders = []
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(data[:, i])
        label_encoders.append(le)

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Навчила регресор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
regressor = ExtraTreesRegressor(n_estimators=100, max_depth=4, random_state=0)
regressor.fit(X_train, y_train)

# Перевірка на невідомій точці
test_point = ['Saturday', '10:20', 'Atlanta', 'no']
encoded_point = []
count = 0
for i, item in enumerate(test_point):
    if item.isdigit():
        encoded_point.append(int(item))
    else:
        encoded_point.append(int(label_encoders[count].transform([item])[0]))
        count += 1

prediction = regressor.predict([encoded_point])
print(f"Спрогнозована інтенсивність руху: {int(prediction[0])}")