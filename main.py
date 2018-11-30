import numpy as np
import preprocessing
import reading
import visualization
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
import copy
import pandas as pd


"""
Test function only
"""
def mainTest():
    numericDataCols = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                       'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                       'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                       'MoSold', 'YrSold']
    for col in numericDataCols:
        trainDF, inputsCol, outputCol = reading.readData()
        outputSeries = (trainDF.loc[:, outputCol] - trainDF.loc[:, outputCol].min()) / (trainDF.loc[:, outputCol].max() - trainDF.loc[:, outputCol].min())
        preprocessing.convertNominalValue(trainDF, outputSeries, inputsCol, outputCol)
        preprocessing.manageNAValues(trainDF, inputsCol)
        preprocessing.testPreprocess(trainDF, trainDF, inputsCol, col)
        # print(trainDF)
        alg = GradientBoostingRegressor(random_state = 1)   # accuracy does not change everytime it is run with set random_state
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        print(np.mean(cvScores))
    # visualization(trainDF,inputsCol,outputCol)


"""
main

Parameters:
    None

Return:
    None
    
Description:
    Main body of the program
"""
def main():
    trainDF, inputsCol, outputCol = reading.readData()
    preprocessing.preprocess(trainDF, trainDF, inputsCol)
    alg = GradientBoostingRegressor(random_state = 1)   # accuracy does not change everytime it is run with set random_state
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    print("Highest Accuracy with all features =", np.mean(cvScores))
    #visualization(trainDF,inputsCol,outputCol)
    
    #Already done, testing what happens to accuracy removing one feature at a time
    """
    inputsColTemp = copy.deepcopy(inputsCol)
    temp = {}
    temp["Nothing removed"] = np.mean(cvScores)
    while len(inputsColTemp) != 0:
        featureRemoved = inputsColTemp.pop()
        inputsCol.remove(featureRemoved)
        alg = GradientBoostingRegressor(random_state = 1)
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        temp[featureRemoved] = np.mean(cvScores)
        print("Accuracy when removing " + featureRemoved + " =", np.mean(cvScores))
        inputsCol.append(featureRemoved)
    export = pd.Series(temp)
    export.to_csv("/Users/zmwilk/Desktop/test.csv")
    """
    
    itemsToRemove = set(["HalfBath", "LandSlope","BldgType","YearBuilt","LowQualFinSF","Utilities","1stFlrSF",
                         "GarageCond","ScreenProch","OpenPorchSF","EnclosedPorch"])
    post_featureRemoval = filter(lambda x: x not in itemsToRemove, inputsCol)
    alg = GradientBoostingRegressor(random_state = 1)
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, post_featureRemoval], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    print("After removing all detrimental features =", np.mean(cvScores))
    #This results in a lower value... does this mean some of these are related, or simply need preprocessing
    #   (e.g., year built should probably become age)?

    #visualization.visualize(trainDF,inputsCol,outputCol)
main()