import numpy as np
import preprocessing
import reading
import visualization
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection


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
    print("Highest Accuracy with all featuers =", np.mean(cvScores))
    #visualization(trainDF,inputsCol,outputCol)
    
    inputsColTemp = inputsCol
    while len(inputsColTemp) != 0:
        featureRemoved = inputsColTemp.pop()
        inputsCol.remove(featureRemoved)
        alg = GradientBoostingRegressor(random_state = 1)
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        print("Accuracy when removing " + featureRemoved + " =", np.mean(cvScores))
        inputsCol.append(featureRemoved)

    visualization.visualize(trainDF,inputsCol,outputCol)
>>>>>>> refs/remotes/DMFinalProject/master
main()