import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
import numpy as np

def readData():
    inputsCol = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
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
       'SaleCondition']
    outputCol = ['SalePrice']
    trainDF = pd.read_csv("data/train.csv")
    return trainDF, inputsCol, outputCol

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

"""
Generic numerical encoder for string values ('a' = 0, 'b' = 1, so on - nothing special)
Original code, based on https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""
def encodeNominalData(inputDF, inputCols):
    reps = len(inputCols)
    for i in range(reps):
        print(i)
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(inputDF.iloc[:,i])
        inputDF.iloc[:,i] = labelEncoder.transform(inputDF.iloc[:,i])
        
def manageNAValues(inputDF, inputCols):
    
    #First, deal with columns where "NA" actually means something
    significantNAFields = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
                           "FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
                           "PoolQC","Fence","MiscFeature"]
    for columnName in significantNAFields:
        inputDF.loc[:,columnName].map(lambda val: "NoValue" if val == "NA" else val)
    
    #Then, fill in remaining "NA" values (currently with mode of other column entries)
    for column in inputCols:
        inputDF.loc[:,column] = inputDF.loc[:,column].fillna(inputDF.loc[:,column].mode()[0])

def preprocess(targetDF, sourceDF):
    exConversionCols = ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    targetDF.loc[:, exConversionCols] = targetDF.loc[:,exConversionCols].applymap(lambda x: nominalValueConversion(x))  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    #targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]] = targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]].applymap(lambda x: 1.0 if x == 'Ex' else (0.75 if x == 'Gd' else (0.5 if x == 'TA' else (0.25 if x == 'Fa' else (0.0 if x == 'Po' else x)))))    # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].map(lambda x: bsmtQualConversion(x))
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].map(lambda x: poolQCConversion(x))


def main():
    trainDF, inputsCol, outputCol = readData()
    #preprocess(trainDF, trainDF)
    manageNAValues(trainDF, inputsCol)
    encodeNominalData(trainDF, inputsCol)
    alg = GradientBoostingRegressor()
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='mean_squared_error')
    print(np.mean(cvScores))
    #print(trainDF.loc[0:10, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
    #                 "PoolQC"]])
    print(trainDF.loc[0:10,:])
main()