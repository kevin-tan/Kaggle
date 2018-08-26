import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing.imputation import Imputer

# Load raw .csv data
raw_test = pd.read_csv('../../data/tiatinic/test/test.csv')
raw_train = pd.read_csv('../../data/tiatinic/train/train.csv')
