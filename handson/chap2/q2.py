import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from scipy.ndimage.interpolation import shift
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data = pd.read_csv('../../data/mnist/digit/train/train.csv')
data_copy = data.copy()
new_data = []
for row in data_copy.iterrows():
    label = row[1].values[0]
    original = row[1].values[1:].copy().reshape(28, 28)

    up = shift(input=original, shift=[-1, 0], cval=0).reshape(1, 784)
    down = shift(input=original, shift=[1, 0], cval=0).reshape(1, 784)
    right = shift(input=original, shift=[0, 1], cval=0).reshape(1, 784)
    left = shift(input=original, shift=[-1, 0], cval=0).reshape(1, 784)

    data.append(pd.Series(np.insert(up, 0, label)), ignore_index=True)
    data.append(pd.Series(np.insert(down, 0, label)), ignore_index=True)
    data.append(pd.Series(np.insert(left, 0, label)), ignore_index=True)
    data.append(pd.Series(np.insert(right, 0, label)), ignore_index=True)

print('Done...')

y = data.label
X = data.drop('label')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)
print(cross_val_score(sgd_clf, X_test, y_test, cv=3, scoring='accuracy'))
