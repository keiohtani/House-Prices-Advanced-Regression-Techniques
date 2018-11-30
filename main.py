import numpy as np
import preprocessing
import reading
import visualization
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
import copy
import pandas as pd
import os


"""
Test function only
"""

class Score:
    def __init__(self, score, name):
        self.score = score
        self.name = name


"""
LandSlope 0.9702540457807078
1stFlrSF 0.9697972747829293
YearBuilt 0.9697369394264912
Original 0.96956577303446
BldgType 0.9694745486085345
GarageCond 0.9693082086453984
ScreenPorch 0.9690054167215087
HalfBath 0.9688091240215666
EnclosedPorch 0.9681940680980062
LowQualFinSF 0.9681576405004482
OpenPorchSF 0.9681087559959425
Utilities 0.9680853759421726
"""


def mainTest():
    scoreList = []
    numericDataCols = ["HalfBath", "LandSlope","BldgType","YearBuilt","LowQualFinSF","Utilities","1stFlrSF",
                         "GarageCond","ScreenPorch","OpenPorchSF","EnclosedPorch"]
    trainDF, inputsCol, outputCol = reading.readData()
    preprocessing.manageNAValues(trainDF, inputsCol)
    preprocessing.preprocess(trainDF, trainDF, inputsCol)
    alg = GradientBoostingRegressor(
        random_state=1)  # accuracy does not change everytime it is run with set random_state
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10,
                                               scoring='r2')
    score = Score(np.mean(cvScores), 'Original')
    scoreList.append(score)
    for col in numericDataCols:
        inputsCol.remove(col);
        trainDF = trainDF.loc[:, inputsCol + [outputCol]];
        alg = GradientBoostingRegressor(random_state = 1)   # accuracy does not change everytime it is run with set random_state
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        score = Score(np.mean(cvScores), col)
        scoreList.append(score)
    # visualization(trainDF,inputsCol,outputCol)
    """
    https://stackoverflow.com/questions/403421/how-to-sort-a-list-of-objects-based-on-an-attribute-of-the-objects
    sorting objects
    """
    scoreList.sort(key=lambda x: x.score, reverse=True)
    for score in scoreList:
        print(score.name, score.score)


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

    # inputsColTemp = copy.deepcopy(inputsCol)
    inputsColTemp = set(["HalfBath", "LandSlope", "BldgType", "YearBuilt", "LowQualFinSF", "Utilities", "1stFlrSF",
                         "GarageCond", "ScreenPorch", "OpenPorchSF", "EnclosedPorch"])
    temp = {}
    temp["Nothing removed"] = np.mean(cvScores)
    # while len(inputsColTemp) != 0:
    #     featureRemoved = inputsColTemp.pop()
    #     inputsCol.remove(featureRemoved)
    #     alg = GradientBoostingRegressor(random_state = 1)
    #     cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    #     temp[featureRemoved] = np.mean(cvScores)
    #     print("Accuracy when removing " + featureRemoved + " =", np.mean(cvScores))
    #     inputsCol.append(featureRemoved)
    # export = pd.Series(temp)
    # export.to_csv(os.getcwd() + '/test.csv')
    # TODO It seems even when deleting one column improves the result, removing the multiple columns worsen the accuracy
    itemsToRemove = set(['OpenPorchSF','YearBuilt', 'ScreenPorch', 'LowQualFinSF', '1stFlrSF', 'EnclosedPorch']) # changed to ScreenPorch from ScreenProch
    post_featureRemoval = filter(lambda x: x not in itemsToRemove, inputsCol)
    alg = GradientBoostingRegressor(random_state = 1)
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, post_featureRemoval], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    print("After removing all detrimental features =", np.mean(cvScores))
    #This results in a lower value... does this mean some of these are related, or simply need preprocessing
    #   (e.g., year built should probably become age)?

    #visualization.visualize(trainDF,inputsCol,outputCol)

main()