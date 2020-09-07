import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("recs2009_public.csv",index_col=0)

#There are many columns showing parameters like "temperature when...", but showing -2 when not aplicable.
#We need to separate this type of data into a binary (not aplicable or aplicable) and a non-categorical column
#In the non-categorical column, we will replace "-2" with the average of all the values different from "-2"

def seriesIntoBinaryAndNonCateg(s,valuesToBinary):
    mean = s[s > 0].mean()
    nonCategSeries = s.replace(valuesToBinary,mean)
    categSeriesList = [(s == value).astype(float) for value in valuesToBinary]
    return pd.concat([nonCategSeries]+categSeriesList,axis=1)

cols_categ_with_binary = set([21,22,26,29,38,41,44,48,54,146,152,310,460,462,466,467,468,540,546,547,548,549,
                          600,602,716,723,776]+list(range(760,773)))
list_all_noncateg_and_binary = list()
for col in cols_categ_with_binary:
    #I found columns with "." that I am assuming to mean "again -2"
    s = pd.to_numeric(df[df.columns[col]].replace('.',-2))
    list_all_noncateg_and_binary.append(seriesIntoBinaryAndNonCateg(s,[-2]))

df_all_noncateg_and_binary = pd.concat(list_all_noncateg_and_binary,axis=1)

#In the case of column "NKRGALNC", 77 means "not sure". thus we have values -2 and 77 to trasnform to binary
#And a non-categorical integer
#in the same way 

s = pd.to_numeric(df[df.columns[717]].replace('.',-2)) #asumming "." is "-2" to save time
df_717 = seriesIntoBinaryAndNonCateg(s,[-2,77])

#In a similar way, columns 595 597 599 601, can be trasnform into a non-categorical column and 3 binary columns 
#corresponding to values -2, -8, -9

cols_noncateg_and_3_binaries = {595,597,599,601}
list_all_noncateg_and_3_binaries = list()
for col in cols_noncateg_and_3_binaries:
    s = pd.to_numeric(df[df.columns[col]].replace('.',-2)) #asumming "." is "-2" to save time
    list_all_noncateg_and_3_binaries.append(seriesIntoBinaryAndNonCateg(s,[-2,-8,-9]))

df_all_noncateg_and_3_binaries = pd.concat(list_all_noncateg_and_3_binaries,axis=1)
    
# We create a list of fully non-categorical columns, as most columns are categorical
cols_full_noncateg=set([4,5,6,7,8,15,30,31,32,33,113,115,117,133,238,288,502,503,556,594,596,598,607,758,759,784] 
                    +list(range(826,836))+list(range(856,906))+list(range(931,939)))
df_full_noncateg=df[df.columns[list(cols_full_noncateg)]]
# The gloal is predicting electricity usage from residential unit information so we remove all columns that
# give direct information about electricity usage, and electricity cost 'KWHSPH'...'DOLELRFG'
cols_to_ignore = set(list(range(839,856))+list(set(range(906,918))))

#the raminig columns correspond to the full categorical ones
cols_full_categ = [col for col in range(len(df.columns)) if col not in cols_categ_with_binary \
                     and col not in cols_full_noncateg and col not in cols_noncateg_and_3_binaries \
                     and col != 717 and col not in cols_to_ignore and col != 838] #838 is the column to be predcited

#We now start with the actual One-hot econding schema
df_categorical = df[df.columns[cols_full_categ]]
X = df_categorical.to_numpy().tolist()
enc = OneHotEncoder()
enc.fit(X)
Y = enc.transform(X).toarray()
df_binary = pd.concat([pd.DataFrame(Y),pd.DataFrame(df.index)],axis =1).set_index("DOEID")

#finally concatenate all the dataframes
df_encoded = pd.concat([df[df.columns[838]],df_all_noncateg_and_binary,df_717,df_all_noncateg_and_3_binaries,df_full_noncateg,df_binary],axis=1)


import numpy as np

msk = np.random.rand(len(df_encoded)) < 0.8
df_training = df_encoded[msk]
df_test = df_encoded[~msk]

testProcentageOfData = len(df_test.index)/(len(df_training.index)+len(df_test.index))*100
traningProcentageofData = 100 - testProcentageOfData
print("{:.2f}".format(testProcentageOfData)+"% of the data correspond to the test set")
print("{:.2f}".format(traningProcentageofData)+"% of the data correspond to the traning set")


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF


df_X = df_training.drop(["KWH"],axis =1) 
df_Y = df_training["KWH"]
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(df_X.to_numpy(), df_Y.to_numpy())
prediction = gpr.predict(df_test.drop(["KWH"],axis =1).to_numpy())
diff = prediction-df_test["KWH"].to_numpy()

print(prediction, len(prediction))

print(df_test["KWH"].to_numpy(), len(df_test["KWH"].to_numpy()))

print(np.std(diff))
print(np.std(df_test["KWH"].to_numpy()))
print(np.std(prediction))

