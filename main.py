import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_row', 1000)

pd.set_option('display.max_columns', 80)

def readData():
    inputsCol = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition'] # I think we should drop Utilities because all the values are same except for one but the result is better with Utilities
    outputCol = 'SalePrice'
    trainDF = pd.read_csv("data/train.csv", usecols=inputsCol + [outputCol])
    #print(trainDF)
    return trainDF, inputsCol, outputCol

def lotShapeValueConversion(x):
    if x == 'Reg':
        return 1.0
    elif x == 'IR1':
        return 0.66
    elif x == 'IR2':
        return 0.33
    elif x == 'IR3':
        return 0.0

def nominalValueConversion(x):  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
     if x == 'Ex':
         return 1.0
     elif x == 'Gd':
         return 0.75
     elif x == 'TA':
        return 0.5
     elif x == 'Fa':
         return 0.25
     elif x == 'Po':
         return 0.0
     else:
         return x

def bsmtQualConversion(x):  # converting the height of the basement
    if x == 'Ex':
        return 105
    elif x == 'Gd':
        return 95
    elif x == 'TA':
        return 85
    elif x == 'Fa':
        return 75
    elif x == 'Po':
        return 65
    else:
        return x

def poolQCConversion(x):    # there is no poor for pool
    if x == 'Ex':
        return 1.0
    elif x == 'Gd':
        return 0.66
    elif x == 'TA':
        return 0.33
    elif x == 'Fa':
        return 0.0
    else:
        return x

def normalization(targetDF, columns):
    targetDF.loc[:, columns] = (targetDF.loc[:, columns] - targetDF.loc[:, columns].min()) / (targetDF.loc[:, columns].max() - targetDF.loc[:, columns].min())

def standardize(df, columns):
    df.loc[:,columns] = (df.loc[:,columns] - df.loc[:,columns].mean()) / df.loc[:,columns].std()

"""
Generic numerical encoder for string values ('a' = 0, 'b' = 1, so on - nothing special)
Original code, based on https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""

def encodeNominalData(inputDF, inputCols):
    reps = len(inputCols)
    for i in range(reps):
        # print(inputCols[i])
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(inputDF.loc[:,inputCols[i]])
        inputDF.loc[:,inputCols[i]] = labelEncoder.transform(inputDF.loc[:,inputCols[i]])
        
def manageNAValues(inputDF, inputCols):
    inputDF.loc[:,'LotFrontage'] = inputDF.loc[:,'LotFrontage'].fillna(0)
    #First, deal with columns where "NA" actually means something
    significantNAFields = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                           "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
                           "PoolQC","Fence","MiscFeature"]
    for columnName in significantNAFields:
        inputDF.loc[:,columnName].map(lambda val: "NoValue" if val == "NA" else val)
    
    #Then, fill in remaining "NA" values (currently with mode of other column entries)
    for column in inputCols:
        inputDF.loc[:,column] = inputDF.loc[:,column].fillna(inputDF.loc[:,column].mode()[0])

def masVnrTypeConversion(x):    # https://www.angieslist.com/articles/how-much-does-brick-veneer-cost.htm
    if x == 'BrkCmn':
        return 15
    elif x == 'BrkFace':
        return 6
    elif x == 'CBlock':
        return 3
    elif x == 'None':
        return 0
    elif x == 'Stone':
        return 30

def fenceValueConversion(x):
    if x == 'GdPrv':
        return 1
    elif x == 'MnPrv':
        return 0.5
    elif x == 'GdWo':
        return 0.75
    elif x == 'MnWw':
        return 0.25
    elif x == 'NA':
        return 0

def preprocess(targetDF, sourceDF, inputsCol):
    outputCol = 'SalePrice'
    exConversionCols = ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    """
    - operation of lists
    https://stackoverflow.com/questions/3428536/python-list-subtraction-operation 
    """

    # outputSeries = (trainDF.loc[:, outputCol] - trainDF.loc[:, outputCol].min()) / (trainDF.loc[:, outputCol].max() - trainDF.loc[:, outputCol].min())
    # convertNominalValue(trainDF, outputSeries, inputsCol)
    nominalDataCol = list(set(inputsCol) - set(exConversionCols) - set(['BsmtQual', 'PoolQC', 'MasVnrType', 'Fence', '1stFlrSF', 'GarageArea'])) # - set(numericDataCols)

    convertNominalValue(targetDF, nominalDataCol, outputCol)
    # encodeNominalData(targetDF, nominalDataCol)
    targetDF.loc[:, exConversionCols] = targetDF.loc[:,exConversionCols].applymap(lambda x: nominalValueConversion(x))  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    #targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]] = targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]].applymap(lambda x: 1.0 if x == 'Ex' else (0.75 if x == 'Gd' else (0.5 if x == 'TA' else (0.25 if x == 'Fa' else (0.0 if x == 'Po' else x)))))    # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].map(lambda x: bsmtQualConversion(x))
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].map(lambda x: poolQCConversion(x))
    targetDF.loc[:, 'MasVnrType'] = targetDF.loc[:, 'MasVnrType'].map(lambda x: masVnrTypeConversion(x))
    targetDF.loc[:, 'Fence'] = targetDF.loc[:, 'Fence'].map(lambda x: fenceValueConversion(x))
    #targetDF.loc[:, "LotShape"] = targetDF.loc[:, "LotShape"].map(lambda x: lotShapeValueConversion(x))
    #standardize(targetDF, inputsCol)  # Accuracy 0.8952159968525466
    normalization(targetDF, inputsCol)  # Accuracy 0.8958124722966672
    # targetDF.loc[:, "YearRemodAdd"] = targetDF.loc[:, ['YearBuilt', "YearRemodAdd"]].apply(lambda row: np.NaN if row.loc['YearBuilt'] == row.loc['YearRemodAdd'] else row.loc['YearRemodAdd'], axis = 1) # remodel year should be adjusted in the case of remodel has not been done.
#
# def preprocess(targetDF, sourceDF, inputsCol, col):     # test function to see the difference between different value to be encoded.
#     exConversionCols = ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual",
#                         "GarageCond"]
#     """
#     - operation of lists
#     https://stackoverflow.com/questions/3428536/python-list-subtraction-operation
#     """
#     print(col)
#     nominalDataCol = list(set(inputsCol) - set(exConversionCols) - set(['BsmtQual', 'PoolQC', 'MasVnrType', 'Fence', '1stFlrSF', 'GarageArea', col]))  # - set(numericDataCols)
#     encodeNominalData(targetDF, nominalDataCol)
#     targetDF.loc[:, exConversionCols] = targetDF.loc[:, exConversionCols].applymap(lambda x: nominalValueConversion(x))  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
#     # targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]] = targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]].applymap(lambda x: 1.0 if x == 'Ex' else (0.75 if x == 'Gd' else (0.5 if x == 'TA' else (0.25 if x == 'Fa' else (0.0 if x == 'Po' else x)))))    # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
#     targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].map(lambda x: bsmtQualConversion(x))
#     targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].map(lambda x: poolQCConversion(x))
#     targetDF.loc[:, 'MasVnrType'] = targetDF.loc[:, 'MasVnrType'].map(lambda x: masVnrTypeConversion(x))
#     targetDF.loc[:, 'Fence'] = targetDF.loc[:, 'Fence'].map(lambda x: fenceValueConversion(x))
#     # targetDF.loc[:, "LotShape"] = targetDF.loc[:, "LotShape"].map(lambda x: lotShapeValueConversion(x))
#     # standardize(targetDF, inputsCol)  # Accuracy 0.8952159968525466
#     normalization(targetDF, inputsCol)  # Accuracy 0.8958124722966672
#     # targetDF.loc[:, "YearRemodAdd"] = targetDF.loc[:, ['YearBuilt', "YearRemodAdd"]].apply(lambda row: np.NaN if row.loc['YearBuilt'] == row.loc['YearRemodAdd'] else row.loc['YearRemodAdd'], axis = 1) # remodel year should be adjusted in the case of remodel has not been done.

def visualization(df, inputsCol, outputCol):
    print('Visualization begins')
    for col in inputsCol:
        # sns.jointplot('SalePrice', col, df)
        # sns.lmplot('SalePrice', col, df)  # draws a line in the graph, I did not see the use
        sns.catplot(x=outputCol, y=col, data=df)  # takes too much time to run
        # sns.swarmplot(x=outputCol, y=col, data=df)    # similar to jointplot but it spreads the points depending on the number of items for each value
        # sns.scatterplot(x=outputCol, y=col, data=df)    # scatterplot does not require values to be numerical so it is good for analysis
        plt.show()
    print("visualization finished")

def convertNominalValue(trainDF, inputCols, outputCol):
    trainDF.loc[:, outputCol] = trainDF.loc[:, 'SalePrice']
    print('Conversion begins')
    colName = 'MSSubClass'  # each column fx. MSSubClass
    aveSeries = trainDF.loc[:, [colName, outputCol]].groupby([colName]).median()
    print(aveSeries)
    print(type(aveSeries.iloc[:, 0]))
    print(aveSeries.iloc[:, 0])
    print(aveSeries.label)
    colSeries = trainDF.loc[:, colName]
    trainDF.loc[:, colName] = trainDF.loc[:, colName].index.map(lambda i: aveSeries.loc[trainDF.loc[:, colName].iloc[i]])
    # for col in inputCols:
    #     # col = 'MSSubClass'  # each column fx. MSSubClass
    #     aveSeries = trainDF.loc[:, [col, outputCol]].groupby([col]).median()
    #     colSeries = trainDF.loc[:, col]
    #     trainDF.loc[:, col] = colSeries.index.map(lambda i: aveSeries.loc[colSeries.iloc[i]])
    print('Conversion ended')
    print(trainDF)



def mainTest():
    numericDataCols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                       'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                       'MoSold', 'YrSold']
    for col in numericDataCols:
        trainDF, inputsCol, outputCol = readData()
    # outputSeries = (trainDF.loc[:, outputCol] - trainDF.loc[:, outputCol].min()) / (trainDF.loc[:, outputCol].max() - trainDF.loc[:, outputCol].min())
    # convertNominalValue(trainDF, outputSeries, inputsCol)
        manageNAValues(trainDF, inputsCol)
    # preprocess(trainDF, trainDF, inputsCol)
        preprocess(trainDF, trainDF, inputsCol, col)
    # print(trainDF)
        alg = GradientBoostingRegressor(random_state = 1)   # accuracy does not change everytime it is run with set random_state
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        print(np.mean(cvScores))
    # visualization(trainDF,inputsCol,outputCol)

def main():
    trainDF, inputsCol, outputCol = readData()
    manageNAValues(trainDF, inputsCol)
    preprocess(trainDF, trainDF, inputsCol)
    print(trainDF)
    alg = GradientBoostingRegressor(random_state = 1)   # accuracy does not change everytime it is run with set random_state
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    print(np.mean(cvScores))
    # visualization(trainDF,inputsCol,outputCol)

main()