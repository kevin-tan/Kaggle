import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

data = pd.read_csv('../../data/creditcard/creditcard.csv')
data.set_index(['Time'], inplace=True)

y = data.Class
X = data.drop(columns=['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



