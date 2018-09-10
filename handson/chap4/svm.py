import pandas as pd
import sklearn.svm as svm
import sklearn.model_selection as ms
import sklearn.metrics as mt

train_data = pd.read_csv('../../data/mnist/digit/train/train.csv')

# Data processing
y = train_data.label
X = train_data.drop(labels=['label'], axis=1)

X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)

# Decision function = one-versus-rest, kernel = Gaussian RBF
svm_clf = svm.LinearSVC(C=0.5)

# Searching for hyperparameter C
def gridSearchForC(svm_clf):
    cv_data = train_data.iloc[:int(train_data.shape[0] * .5), :]
    y = cv_data.label
    X = cv_data.drop(labels=['label'], axis=1)
    param = [{'C': [0.5, 0.6, 0.7, 0.8]}]
    grid_search = ms.GridSearchCV(svm_clf, scoring='accuracy', cv=3, param_grid=param)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)


svm_clf.fit(X_train, y_train)
preds = svm_clf.predict(X_test)
print(mt.accuracy_score(y_test, preds))