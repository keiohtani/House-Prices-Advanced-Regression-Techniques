import pandas as pd

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

def preprocess(targetDF, sourceDF):
    exConversionCols = ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    targetDF.loc[:, exConversionCols] = targetDF.loc[:,exConversionCols].applymap(lambda x: nominalValueConversion(x))  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    #targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]] = targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]].applymap(lambda x: 1.0 if x == 'Ex' else (0.75 if x == 'Gd' else (0.5 if x == 'TA' else (0.25 if x == 'Fa' else (0.0 if x == 'Po' else x)))))    # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].map(lambda x: bsmtQualConversion(x))
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].map(lambda x: poolQCConversion(x))


def main():
    trainDF, inputsCol, outputCol = readData()
    preprocess(trainDF, trainDF)
    print(trainDF.loc[0:10, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond",
                     "PoolQC"]])
main()