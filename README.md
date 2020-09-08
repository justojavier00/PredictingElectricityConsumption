# Introduction

The goal of this repository is to find a regression approach that can be used to predict electricity consumption from residential unit and household data. We have extracted the information from the [2009 version of the RECS program](https://www.eia.gov/consumption/residential/data/2009/index.php?view=microdata).

We start with the data processing step, then the creation of a training set and a test set, then use two selected regression frameworks, and end with a few concluding remarks.
Find the code and results
below or in the [original
notebook](https://github.com/justojavier00/PredictingElectricityConsumption/blob/master/Code/ForecatingElectricityConsumption.ipynb).

## Data Processing

Many columns of the table have categorical data.
We use One-hot encoding schema and another approach, as will be explained below, to convert this data to binary.

Data leakage is prevented by removing columns that are directly related to the electricity consumption (KWH). 
Specifically, columns from 'KWHSPH' to 'DOLELRFG' are directly related to electricity usage and electricity cost.
Also, columns from 'TOTALBTU' to 'TOTALDOLOTH' are composed of total energy consumption of different types. 
The presence of these columns in the data gives the impression of abnormally high precision.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Data/recs2009_public.csv",index_col=0)


# There are many columns showing parameters like "temperature when...", but showing -2 when not applicable.
# We need to separate this type of data into a binary (not applicable or applicable) and a non-categorical column
# In the non-categorical column, we will replace "-2" with the average of all the values larger than 0

def seriesIntoBinaryAndNonCateg(s,valuesToBinary):
    mean = s[~s.isin(valuesToBinary)].mean()
    nonCategSeries = s.replace(valuesToBinary,mean)
    categSeriesList = [(s == value).astype(float) for value in valuesToBinary]
    return pd.concat([nonCategSeries]+categSeriesList,axis=1)

cols_categ_with_binary = set([21,22,26,29,38,41,44,48,54,146,152,310,460,462,466,467,468,540,546,547,548,549,
                          600,602,716,723,776]+list(range(760,773)))

list_all_noncateg_and_binary = list()
for col in cols_categ_with_binary:
    # I found columns with "." that I am assuming to mean "again -2"
    s = pd.to_numeric(df[df.columns[col]].replace('.',-2))
    list_all_noncateg_and_binary.append(seriesIntoBinaryAndNonCateg(s,[-2]))
df_all_noncateg_and_binary = pd.concat(list_all_noncateg_and_binary,axis=1)

# In the case of column "NKRGALNC", 77 means "not sure". thus we have values -2 and 77 to trasnform to binary
# And a non-categorical integer

s = pd.to_numeric(df[df.columns[717]].replace('.',-2)) # Asumming "." is "-2" to save time
ds_717 = seriesIntoBinaryAndNonCateg(s,[-2,77])

# In a similar way, columns 595 597 599 601, can be trasnform into a non-categorical column and 3 binary columns 
# corresponding to values -2, -8, -9.

cols_noncateg_and_3_binaries = {595,597,599,601}
list_all_noncateg_and_3_binaries = list()
for col in cols_noncateg_and_3_binaries:
    s = pd.to_numeric(df[df.columns[col]].replace('.',-2)) # Asumming "." is "-2" to save time
    list_all_noncateg_and_3_binaries.append(seriesIntoBinaryAndNonCateg(s,[-2,-8,-9]))

df_all_noncateg_and_3_binaries = pd.concat(list_all_noncateg_and_3_binaries,axis=1)
    
# We create a list of fully non-categorical columns, as most columns are categorical
cols_full_noncateg=set([4,5,6,7,8,15,30,31,32,33,113,115,117,133,238,288,502,503,556,594,596,598,607,758,759,784] 
                    +list(range(826,836))+list(range(856,906))+list(range(931,939)))
df_full_noncateg=df[df.columns[list(cols_full_noncateg)]]

# The gloal is predicting electricity usage from residential unit information so we remove all columns that
# give direct information about electricity usage, and electricity cost (from 'KWHSPH' to 'DOLELRFG'). 
# Also, we have removed columns reflecting total energy consumption (from 'TOTALBTU' to 'TOTALDOLOTH').
cols_to_ignore = set(list(range(839,856))+list(set(range(906,918))))

# The raminig columns correspond to the full categorical ones.
cols_full_categ = [col for col in range(len(df.columns)) if col not in cols_categ_with_binary \
                     and col not in cols_full_noncateg and col not in cols_noncateg_and_3_binaries \
                     and col != 717 and col not in cols_to_ignore and col != 838] #838 is the column to be predcited

# We now start with the actual One-hot econding schema.
df_categorical = df[df.columns[cols_full_categ]]
X = df_categorical.to_numpy().tolist()
enc = OneHotEncoder()
enc.fit(X)
Y = enc.transform(X).toarray()
df_binary = pd.DataFrame(Y, index=df.index)

# Finally concatenate all the dataframes.
df_encoded = pd.concat([df[df.columns[838]],df_all_noncateg_and_binary,ds_717,df_all_noncateg_and_3_binaries,df_full_noncateg,df_binary],axis=1)

```

## Creating a training set and a test set

We create a training and a test set, using a random approach.
We will only use the test set when the models are complete in order to validate it


```python
import numpy as np
from sklearn.model_selection import train_test_split

X = df_encoded.drop(["KWH"],axis =1).to_numpy()
y = df_encoded["KWH"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

```

## Random Forest Classifier

Because of the nature of the data, we expect it to be stochastic to some extent. 
We start using random forest because of its simplicity.


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

clf = RandomForestClassifier(n_estimators = 250,max_depth=3,random_state=0, criterion="entropy")
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
print("Standard Deviation of Diff:",np.std(prediction-y_test))     # See conclusions
print("Standard Deviation of y_test:", np.std(y_test))             # See conclusions
print("Root Mean Square Error: ",mean_squared_error(y_test, prediction, squared=False)) # Returns RMSE
```

    Standard Deviation of Diff: 5931.193861561492
    Standard Deviation of y_test: 7107.848939035486
    Root Mean Square Error:  6386.42650274541


## Gaussian Process Regressor

Another method that is effective with stochastically distributed variables is the gaussian process regressor


```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.metrics import mean_squared_error

kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train, y_train)

prediction = gpr.predict(X_test)
print("Standard Deviation of Diff:",np.std(prediction-y_test))     # See conclusions
print("Standard Deviation of y_test:", np.std(y_test))             # See conclusions
print("Root Mean Square Error: ",mean_squared_error(y_test, prediction, squared=False)) # Returns RMSE

```

    Standard Deviation of Diff: 5001.419411073517
    Standard Deviation of y_test: 7107.848939035486
    Root Mean Square Error:  5005.4443014970175


## Conclusions


We use the root mean square error as a measure of the error of the prediction, and the standard deviation of the difference between the true values and the predicted corresponding values as a sanity check. This is, if the standard deviation of 'prediction-y_test' is lower than the standard deviation of 'y_test', the prediction is being successful to some extent.

Both the random forest classifier and the Gaussian process regressor predict to some extent the electricity consumption. The latter is considerably better at making this prediction. However, none of the two models is exceptionally good and further exploration is necessary. These two models may serve as the
first two benchmark models to eventually obtain an optimal solution.


```python

```
