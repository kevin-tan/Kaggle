import pandas as pd
import json

data = pd.read_csv('../../../data/comp/train.csv')

# Expand feature column with sub-columns from totals
t = json.loads(data.totals.iloc[0])
totals_expanded = [('totals_' + key, []) for key in t.keys()]
print(totals_expanded)

# Func to parse totals column and expanding df
def expandedTotals(df):
    totals_df = pd.DataFrame()

    return df.join(totals_df)
