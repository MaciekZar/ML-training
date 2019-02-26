import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#import data
df = pd.read_csv()
# dummie variable
col = pd.get_dummies(df[''],drop_first=True)
df.drop('', axis=1, inplace=True)
df = pd.concat([col,df],axis=1)
print(df)
print('=='*70)

#test and train data
X_train, X_test, y_train, y_test = train_test_split(x,y   , test_size=0.2, random_state=0)

#linear regression object
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

#simples table with results
df3 = pd.DataFrame()
df3['y_pred'] = y_pred
df3['y_result'] = y_test
df3['difference'] = y_test - y_pred
print(df3)

#Building best model

#
#start backwardElimination
def backwardElimination(x,y, sl, columns = False):
    """Automatic search for best features set BASED on p-value
    x- pandas.DataFrame y = list/column , sl - alfa
    columns=False -  return numpy array with best features
    column=True - return name of columns
    """
    X = x.values
    numVars = len(X[0])
    for i in range(0, numVars):
        regressor_OLS = smf.OLS(y, X).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    X = np.delete(X, j, 1)
    print(regressor_OLS.summary())
    X_modeled_columns = pd.DataFrame(X)
    valid_features = [column for column in x.columns for col in X_modeled_columns.columns if x[column].equals(X_modeled_columns[col])]
    print(valid_features)
    if columns == True:
        return valid_features
    return x
# end backwardElimination

# example, alfa = 0.05
SL = 0.05
X_opt = pd.concat([pd.Series([1 for n in range(len(df.index))]),df.loc[:,[ ''  ]]],axis=1).values
X_Modeled = backwardElimination(X_opt,df[''], SL)
print(X_Modeled)

#To samo tylko ręcznie
# df2 = pd.concat([pd.Series([1 for row in df['']]),df.loc[:,['' ]]],axis=1)
# regressor_stats = smf.OLS(df[''], df2).fit()
# print(regressor_stats.summary())
# print('=='*70)




