import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

# Load .csv file
data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Setup train_X/train_Y, test_X/test_Y data
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = data[features]
y = data.SalePrice
train_X, test_X, train_y, test_y=train_test_split(X, y) # Splitting data into train and test data

# DecisionTreeRegressor
def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return mean_absolute_error(test_y, preds)

# Optimize model predictions by finding max leaf nodes for DecisionTreeRegressor
test_nodes = [5,25,50,100,250,500]

# Get node size with smallest MAE
nodes_size_to_mae = {get_mae(nodes, train_X, test_X, train_y, test_y): nodes for nodes in test_nodes}
max_node_size = nodes_size_to_mae[min(nodes_size_to_mae.keys())]

# Fit to model
decision_tree = DecisionTreeRegressor(max_leaf_nodes=max_node_size, random_state=0)
decision_tree.fit(train_X, train_y)
decision_tree_pred = decision_tree.predict(test_X)
print("DecisionTree MAE:", mean_absolute_error(test_y, decision_tree_pred))

#RandomForestRegressor
random_forest = RandomForestRegressor(random_state=0)
random_forest.fit(train_X, train_y)
random_forest_pred = random_forest.predict(test_X)
print("RandomForest MAE:", mean_absolute_error(test_y, random_forest_pred))

# Handling missing values
missing_vals = data.isnull().sum()
col_missing_vals = missing_vals[missing_vals > 0]

# Option 1: Drop all cols with missing vals (Not the best option)
# cols_with_missing = [col for col in data.columns if data[col].isnull().any()]
# reduced_data = data.drop(cols_with_missing, axis=1) # row=0, col=1
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

# One-hot encoding (Categorical Data)
# Use the pd.get_dummies() on the categorical columns

from xgboost import XGBRegressor
# XGBoost