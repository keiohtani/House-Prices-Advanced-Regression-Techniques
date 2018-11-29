import pandas as pd

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 80)

"""
readData

Parameters:
    None

Return:
    - trainDF = DataFrame corresponding with input data values for model training
    - inputsCol = list of input columns (features)
    - outputCol = name of class variable column

Description:
    Reads data in from provided csv file and organizes data for data analysis and model training
"""


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
                 'SaleCondition']  # We may want to drop Utilities because all the values are same except for one but the result is better with Utilities
    outputCol = 'SalePrice'
    trainDF = pd.read_csv("data/train.csv", usecols=inputsCol + [outputCol])
    return trainDF, inputsCol, outputCol