import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

# Load .csv file
data = pd.read_csv("../data/houses/train.csv")
test_data = pd.read_csv("../data/houses/test.csv")

# Setup train_X/train_Y, test_X/test_Y data
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data[features]
y = data.SalePrice
train_X, test_X, train_y, test_y = train_test_split(X, y)  # Splitting data into train and test data


# DecisionTreeRegressor
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return mean_absolute_error(test_y, preds)


# Optimize model predictions by finding max leaf nodes for DecisionTreeRegressor
test_nodes = [5, 25, 50, 100, 250, 500]

# Get node size with smallest MAE
nodes_size_to_mae = {get_mae(nodes, train_X, test_X, train_y, test_y): nodes for nodes in test_nodes}
max_node_size = nodes_size_to_mae[min(nodes_size_to_mae.keys())]

# Fit to model
decision_tree = DecisionTreeRegressor(max_leaf_nodes=max_node_size, random_state=0)
decision_tree.fit(train_X, train_y)
decision_tree_pred = decision_tree.predict(test_X)
print("DecisionTree MAE:", mean_absolute_error(test_y, decision_tree_pred))

# RandomForestRegressor
random_forest = RandomForestRegressor(random_state=0)
random_forest.fit(train_X, train_y)
random_forest_pred = random_forest.predict(test_X)
print("RandomForest MAE:", mean_absolute_error(test_y, random_forest_pred))

# -------------------------------------------------------------------------------------------------

# Handling missing values
missing_vals = data.isnull().sum()
col_missing_vals = missing_vals[missing_vals > 0]

# Option 1: Drop all cols with missing vals (Not the best option)
# cols_with_missing = [col for col in data.columns if data[col].isnull().any()]
# reduced_data = data.drop(cols_with_missing, axis=1) # index=0 (since index are down), column label=1 (since column are across)
# axis = 0 = down, axis =1 = across
# reduced_test_data = test_data.drop(cols_with_missing, axis=1)

# Option 2: Imputation (Better option)
# imputer = Imputer()
# data_imputed = imputer.fit_transform(data)

# Option 3: Extension to Imputation (Some case improves results)
# new_data = pd.DataFrame(data.copy())
# cols_with_missing = [col for col in new_data.columns if new_data[col].isnull().any()]
# Adding extra column indicating which will be imputed
# for col in cols_with_missing:
#    new_data[col + "_was_missing_a_val"] = new_data[col].isnull() # Adding new column called col + "_was_misisng_a_val"
# Etc ...

# -------------------------------------------------------------------------------------------------

# One-hot encoding (Categorical Data)
# Use the pd.get_dummies() on the categorical columns

# -------------------------------------------------------------------------------------------------

# XGBoost (model which works well with tabular data)
from xgboost import XGBRegressor

data.dropna(axis=0, subset=['SalePrice'], inplace=True)  # Remove any SalePrice row containing null
X1 = data.drop(['SalePrice'], axis=1).select_dtypes(
    exclude=['object'])  # Get all cols except sale price and object types
y1 = data.SalePrice  # Get sale price col
train_X1, test_X1, train_y1, test_y1 = train_test_split(X1.values, y1.values,
                                                        test_size=0.25)  # Reserve 25% of data as test

# Impute missing data
imputer = Imputer()
train_X1 = imputer.fit_transform(train_X1)
test_X1 = imputer.transform(test_X1)

# n_estimators: how many time we want to go through the estimation cycle
# learning_rate: can help reduce overfitting (good against test data, horrible in prediction) by each tree we add to ensemble helping us less
# Small learning rate + large estimator = more accurate XGBoost models (takes longer since does more iterations)
# Using parallelism to build models faster, use n_jobs (usually set to the # of cores). Good for large datasets
xgbModel = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# early_stopping_rounds: number of consecutive rounds of deterioration of validation score is allowed before stopping
# Eval set is a list of tuple used as validation set for early stopping
# Can take back the test data after finding the optimal n_estimators and train the model with 100% of data
xgbModel.fit(train_X1, train_y1, early_stopping_rounds=5, eval_set=[(test_X1, test_y1)], verbose=False)
print(xgbModel.best_iteration)
prediction = xgbModel.predict(test_X1)
print("XGBoost MAE:", mean_absolute_error(prediction, test_y1))

# -------------------------------------------------------------------------------------------------

# Partial Dependence plot
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import matplotlib.pyplot as plt

col_to_plot = ['LotArea', 'YearBuilt', 'OverallQual']


def get_some_data():
    data = pd.read_csv("../data/houses/train.csv")
    y = data.SalePrice
    X = data[col_to_plot]
    imputer = Imputer()
    imputed_X = imputer.fit_transform(X)
    return imputed_X, y


X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
# Partial Dependence Plotting
# features: specify the index of which of the feature_names you want to plot
# feature_names: list of columns you want to see
plot_partial_dependence(my_model, features=[0, 2], X=X, feature_names=col_to_plot, grid_resolution=50)
# plt.show()

# -------------------------------------------------------------------------------------------------

# Pipeline (making the data processing code clean)
from sklearn.pipeline import make_pipeline

# Always must start with a transformer and end with a model
# Transformers are pre-processing before the modeling, ex. Imputer, there can be a sequence of transformers in a pipeline
# Models are used for the prediction. Transformers are usually pre-processing data before getting fitted into the model
pipeline = make_pipeline(Imputer(), RandomForestRegressor())

# The next two lines are equivalent to imputing the data for train_X and test_X and then fitting and predicting
pipeline.fit(train_X, train_y)
print("Pipeline with RFR MAE:", mean_absolute_error(test_y, pipeline.predict(test_X)))

# -------------------------------------------------------------------------------------------------

# Cross-Validation (more reliable than using train_test_split but longer process to measure model's quality)
# It is the processing of running our model against different subsets of the data (splitting up the data differently for train and test)
# This gives us multiple measures of model quality
# Ex. In a 5 fold/experiment, we divide the data into 5 where 4/5 is the training data and 1/5 is the test.
# The 1st experiment will use the first 20% of the dataset as test data, and the rest as train data
# The 2nd experiment will use the next 20% of the dataset and so on.

# Usually for small dataset use cross-validation
# and large dataset use train_test_split
# But always make sure considering both, time vs. quality trade-off

data = pd.read_csv("../data/houses/train.csv")
col_to_use = ["LotArea", "OverallQual", "GarageArea", "TotRmsAbvGrd"]
X = data[col_to_use]
y = data.SalePrice

pipeline = make_pipeline(Imputer(), RandomForestRegressor())
scoring = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error')
print(f'Mean MAE: {-1*scoring.mean()}')  # String interpolation f'... {vars + operations possible}'

# Data leakage
# What is it? Data leakage causes the model to look accurate but when its time to make decisions with the model, it becomes very inaccurate.

# Two types: Leaky Predictors and Leaky Validation Strategies

# Leaky Predictors
# Occures when predictors include data that will not be available when a prediction is being done
# Prevention is data specific, two rule of thumb:
# > Look for possible leaky predictors, look for columns that are statistically correlated to the target
# > If your model is extremely accurate, possible leaky predictor

# Leaky Validation Strategy
# Occurs when we aren't careful distinguishing between validation data and training data.
# Ex. Imputating the data before separating the train and test data.
# Which corrupts the validation data since Validation is meant to be a measure of how the model does on data that it hasn't considered yet.
# Preventions:
# Use Pipelines to reduce this possible error (critical that Pipelines are used in cross-validation)
