import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

# Fetch MNIST digit data
data = pd.read_csv('../../data/mnist/digit/train/train.csv')


# Function to get separate features and target data
def getTrainingTargetSet(data):
    y = data.label
    X = data.drop(columns=['label'])
    return X, y


# Data preparation
X_train, y_train = getTrainingTargetSet(data.iloc[:28000])
X_validation, y_validation = getTrainingTargetSet(data.iloc[28000:35000])
X_test, y_test = getTrainingTargetSet(data.iloc[35000:])


# Cold turkey classifier
def trainNonTunedClassifiers():
    # Random Forest Classifier
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_validation)
    print('rf_pred', accuracy_score(y_validation, rf_pred))

    # Support Vector Machine Classifier
    svm_clf = LinearSVC()
    svm_clf.fit(X_train, y_train)
    svm_pred = svm_clf.predict(X_validation)
    print('svm_pred', accuracy_score(y_validation, svm_pred))

    # Extra-Trees Classifier
    et_clf = ExtraTreesClassifier()
    et_clf.fit(X_train, y_train)
    et_pred = et_clf.predict(X_validation)
    print('et_pred:', accuracy_score(y_validation, et_pred))

    # SGDClassifier
    sgd_clf = SGDClassifier()
    sgd_clf.fit(X_train, y_train)
    sgd_pred = sgd_clf.predict(X_validation)
    print('sgd_pred:', accuracy_score(y_validation, sgd_pred))

    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    dt_pred = dt_clf.predict(X_validation)
    print('dt_pred:', accuracy_score(y_validation, dt_pred))


# Voting Classifier
def votingClassifier():
    hard_voting_clf = VotingClassifier(
        estimators=[('rf', RandomForestClassifier()), ('et', ExtraTreesClassifier()),
                    ('sg', SGDClassifier())])
    hard_voting_clf.fit(X_train, y_train)
    print(hard_voting_clf.score(X_validation, y_validation))

    hard_voting_clf.set_params(sg=None)
    del hard_voting_clf.estimators_[2]
    hard_voting_clf.voting = 'soft'
    print(hard_voting_clf.score(X_validation, y_validation))


# trainNonTunedClassifiers()
votingClassifier()

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_validation)

# Extra-Trees Classifier
et_clf = ExtraTreesClassifier()
et_clf.fit(X_train, y_train)
et_pred = et_clf.predict(X_validation)

# SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train)
sgd_pred = sgd_clf.predict(X_validation)

data = np.array([rf_pred, et_pred, sgd_pred])
# Stacking ensemble
stacking_data = pd.DataFrame(columns=['rd_pred', 'et_pred', 'sgd_pred'],
                             data=data.T)
blender = RandomForestClassifier()
blender.fit(stacking_data, y_validation)


def predict(x):
    d = np.array([rf_clf.predict(x), et_clf.predict(x), sgd_clf.predict(x)])
    s_d = pd.DataFrame(columns=['rd_pred', 'et_pred', 'sgd_pred'],
                       data=d.T)
    return blender.predict(s_d)


print(accuracy_score(y_test, predict(X_test)))
