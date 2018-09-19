import numpy as np
import pandas as pd
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.preprocessing as pre
import scipy.stats as sp
import sklearn.metrics as mt

train_data = pd.read_csv('../../../data/mnist/digit/train/train.csv')

# Data processing
y = train_data.label
X = train_data.drop(labels=['label'], axis=1)

standard = pre.StandardScaler()
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)

X_train_scaled = standard.fit_transform(X_train.astype(np.float32))
X_test_scaled = standard.transform(X_test.astype(np.float32))

# Decision function = one-versus-rest, kernel = Gaussian RBF
svm_clf = svm.SVC(C=7.649368301467556, gamma=0.001291535035355349) # ~96%


# Searching for hyperparameter C
def gridSearchForC(svm_clf):
    cv_data = train_data.iloc[:1000, :]
    y = cv_data.label
    X = cv_data.drop(labels=['label'], axis=1)
    X = standard.fit_transform(X.astype(np.float32))
    param = {'C': sp.uniform(1, 10), 'gamma': sp.reciprocal(0.001, 0.1)}
    rand_search = ms.RandomizedSearchCV(svm_clf, param, n_iter=10)
    rand_search.fit(X, y)
    print(rand_search.best_estimator_)


# gridSearchForC(svm_clf)

svm_clf.fit(X_train_scaled, y_train)
preds = svm_clf.predict(X_test_scaled)
print(mt.accuracy_score(y_test, preds))
