import pandas as pd
from sklearn.neighbors import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import *
from sklearn.linear_model import *

data = pd.read_csv('../../data/mnist/digit/train/train.csv')

# Preparing data
data.dropna(axis=0, inplace=True)

y = data.label
X = data.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = [
    {'n_neighbors': [5, 6, 7, 8], 'weights': ['uniform', 'distance']}
]

knn_clf = KNeighborsClassifier()

grid = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)
print(grid.best_estimator_)
