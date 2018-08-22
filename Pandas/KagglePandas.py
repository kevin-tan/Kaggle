import pandas as pd

pd.set_option('max_rows', 5)

# DataFrame
# Index is the row names
# The data is in the form "column": [data corresponding to an index]
data = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]}, index=["2017 sale", "2018 sale"])
print(data)

print()

data = pd.DataFrame({'HasPhone': [True, True, False], 'Person': ["Kevin", "Fatima", "Boosted"]})
print(data, "\n")
print(data.Person[data.HasPhone])
print("\n", data.Person[~data.HasPhone], "\n")

data = pd.DataFrame({'HasPhone': ["iPhone", "iPhone", "Android"], 'Person': ["Kevin", "Fatima", "Boosted"]})
print(data, "\n")
print(data.loc[:1, ["Person"]], "\n")  # .loc([index], [col]), iloc() excludes upper-bound
# Indexing with Boolean (only returns the rows with True)
criteria = data.HasPhone.map(lambda x: x is "iPhone")
print(data.Person[criteria], "\n")
print(data.Person[data.HasPhone.isin(["iPhone"])], "\n")

# Series
s = pd.Series([4, 2, 2, 1], index=["Flour", "Milk", "Eggs", "Spam"], name="Dinner")
print(s, "\n")
print(s.loc["Eggs":], "\n")
print(s.loc[:"Eggs"])
# Indexing with Boolean
print(s[s > 2], "\n")
print(s[s.isin([2, 4])])

# We can divide two series which returns a mutated series with all its values conforming to the division i.e. (series.A / series.B).values.idxmax() returns the position of the max val of the ratio between A and B
# Can use value_counts() returning the total counts of unique values with row label as its unique value
# Operations on series will return a series, i.e. +, -, *, / ,...

# Important to know DataFrame, Series, GroupBy (DataFrameGroupBy, SeriesGroupBy)
