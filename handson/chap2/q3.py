import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score

# Get data from .csv file
data = pd.read_csv('../../data/tiatinic/train/train.csv')


# Encoder function
def encodeText(data, col, encoder=OneHotEncoder()):
    encoded, categories = data[col].factorize()
    # We want a Kx1 matrix where K is the number of instances
    values_encoded = encoder.fit_transform(encoded.reshape(-1, 1))
    data.drop(columns=[col], inplace=True)
    temp_df = pd.DataFrame(values_encoded.toarray(), columns=categories)
    return pd.concat([data, temp_df], axis=1)  # to append on the features


# Separate data into test and train data set
y = data.Survived
X = data.drop(columns=['Survived', 'Name', 'Ticket', 'Embarked', 'PassengerId'])

# Clean data
X.Cabin.fillna('NA', inplace=True)
X = encodeText(X, 'Sex')
X = encodeText(X, 'Cabin')
imputer = Imputer()
# Split data before imputing (only want to impute on train data to prevent data snooping bias)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# Testing with SDGClassifier
sdg_clf = SGDClassifier()
sdg_perf = cross_val_score(sdg_clf, X_train_imputed, y_train, scoring='accuracy', cv=3)
print('sdg_perf', sdg_perf)

# Testing with LogisticRegression
lgs_reg = LogisticRegression(C=0.5)
lgs_perf = cross_val_score(lgs_reg, X_train_imputed, y_train, scoring='accuracy', cv=3)
print('lgd_perf', lgs_perf)

# Testing with RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth=10)
rf_perf = cross_val_score(rf_clf, X_train_imputed, y_train, scoring='accuracy', cv=3)
print('rf_perf', rf_perf)


def getBestHyperparameterForEstimator():
    rf_param = [{'max_depth': [5, 6, 7, 8, 9, 10]}]
    grid_search = GridSearchCV(rf_clf, rf_param, scoring='accuracy', cv=3)
    grid_search.fit(X_train_imputed, y_train)
    print(grid_search.best_estimator_)

# Analyze data
def graphing_analysis(X_train, y_train):
    X_train_graph = X_train.copy()
    X_train_graph['Survived'] = y_train
    # More females survived than males, highly correlated
    fix, axarr = plt.subplots(1, 2)
    X_train_graph.Sex[X_train_graph.Survived == 1].value_counts().plot.bar(ax=axarr[0], title='Survived')
    X_train_graph.Sex[X_train_graph.Survived == 0].value_counts().plot.bar(ax=axarr[1], title='Dead')
    plt.show()

    # Ages are around the same but we can see younger age tends to survive
    fix2, axarr2 = plt.subplots(1, 2)
    sns.kdeplot(X_train_graph.Age[X_train_graph.Survived == 1].value_counts().sort_index(), ax=axarr2[0]).set_title(
        'Survived')
    sns.kdeplot(X_train_graph.Age[X_train_graph.Survived == 0].value_counts().sort_index(), ax=axarr2[1]).set_title(
        'Dead')
    plt.show()

    # Strongly correlated, for pclass = 1 they mostly survived where pclass = 3 most of them didn't
    fix3, axarr3 = plt.subplots(1, 2)
    X_train_graph.Pclass[X_train_graph.Survived == 1].value_counts().plot.bar(ax=axarr3[0], title='Survived')
    X_train_graph.Pclass[X_train_graph.Survived == 0].value_counts().plot.bar(ax=axarr3[1], title='Dead')
    plt.show()

    # It seems that fare with smaller cost survived
    fix4, axarr4 = plt.subplots(1, 2)
    sns.kdeplot(X_train_graph.Fare[X_train_graph.Survived == 1].value_counts().sort_index(), ax=axarr4[0]).set_title(
        'Survived')
    sns.kdeplot(X_train_graph.Fare[X_train_graph.Survived == 0].value_counts().sort_index(), ax=axarr4[1]).set_title(
        'Dead')
    plt.show()

    # Parch should have little to no relevance to prediction
    fix5, axarr5 = plt.subplots(1, 2)
    X_train_graph.Parch[X_train_graph.Survived == 1].value_counts().plot.bar(ax=axarr5[0], title='Survived')
    X_train_graph.Parch[X_train_graph.Survived == 0].value_counts().plot.bar(ax=axarr5[1], title='Dead')
    plt.show()
