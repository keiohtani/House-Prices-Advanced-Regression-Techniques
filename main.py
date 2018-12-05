import numpy as np
import preprocessing
import reading
import visualization
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
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
    preprocessing.preprocess(trainDF, trainDF, inputsCol)
    alg = GradientBoostingRegressor(
        random_state=1)  # accuracy does not change everytime it is run with set random_state
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    score = Score(np.mean(cvScores), 'Original')
    scoreList.append(score)
    for col in numericDataCols:
        inputsCol.remove(col);
        print(inputsCol)
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
    print("Highest Accuracy with all features, default parameterizations =", np.mean(cvScores))
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
    export.to_csv(os.getcwd() + '/removeOneFeature_postDateToAgeConversion.csv')
    """
    
    #Testing various parameterizations
    '''
    print("Changing n:")
    i = 100
    while i<=1000:
        alg = GradientBoostingRegressor(random_state = 1, n_estimators = i)
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        print("For score of i = " + str(i) + ":", np.mean(cvScores))
        i += 100
    '''
    '''
    print("Changing learning rate:")
    j = 0.01
    while j<=1:
        alg = GradientBoostingRegressor(random_state = 1, learning_rate = j)
        cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
        print("For rate of j = " + str(j) + ":", np.mean(cvScores))
        j += 0.05
    '''
    """
    print("Testing proposed optimum settings (approximate):")
    alg = GradientBoostingRegressor(random_state = 1, n_estimators = 900, learning_rate = 0.16)
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    print("Score:", np.mean(cvScores))
    """
    """
    For score of i = 100: 0.9695838362322334
    For score of i = 200: 0.9713926415403247
    For score of i = 300: 0.97218152251929
    For score of i = 400: 0.9725433964858627
    For score of i = 500: 0.9727129864730497
    For score of i = 600: 0.9727150286009545
    For score of i = 700: 0.9727550478839996
    For score of i = 800: 0.9728148128730638
    For score of i = 900: 0.9727879075157484
    For score of i = 1000: 0.9727582470818399
    """
    
    """
    Changing learning rate:
    For rate of j = 0.01: 0.7866244576707975
    For rate of j = 0.060000000000000005: 0.9673647242443091
    For rate of j = 0.11000000000000001: 0.9685577086455324
    For rate of j = 0.16000000000000003: 0.9706671931416834
    For rate of j = 0.21000000000000002: 0.9704141690215575
    For rate of j = 0.26: 0.9693811639251051
    For rate of j = 0.31: 0.9640193537044937
    For rate of j = 0.36: 0.9662721202679398
    For rate of j = 0.41: 0.9650110713531361
    For rate of j = 0.45999999999999996: 0.9598916595713343
    For rate of j = 0.51: 0.9612472671240511
    For rate of j = 0.56: 0.9582289549106051
    For rate of j = 0.6100000000000001: 0.9532347135301027
    For rate of j = 0.6600000000000001: 0.9568633720893616
    For rate of j = 0.7100000000000002: 0.951672708291688
    For rate of j = 0.7600000000000002: 0.9524549341492469
    For rate of j = 0.8100000000000003: 0.9461558271946371
    For rate of j = 0.8600000000000003: 0.9354096980604721
    For rate of j = 0.9100000000000004: 0.9328281792809602
    For rate of j = 0.9600000000000004: 0.9325540177700384
    """
    
    #Optimum for both at the same time = 0.9708920005637681... interesting... so looks like better to focus on n_estimators

    #Testing combinations of removing items
    # without ["HalfBath", "LandSlope", "BldgType", "YearBuilt", "LowQualFinSF", "Utilities"] 0.9697767256899044
    # without ["HalfBath", "LandSlope", "BldgType", "YearBuilt", "LowQualFinSF"] 0.9689720298816091
    # without ["HalfBath", "LandSlope", "BldgType", "YearBuilt"] 0.9689965270724571
    # without ["HalfBath", "LandSlope", "BldgType"] 0.9701461139168301
    # without ["HalfBath", "LandSlope"] 0.968366504937428
    # without ["HalfBath"] 0.968366504937428
    # TODO It seems even when deleting one column improves the result, removing the multiple columns worsen the accuracy

    #Test function for above accuracies
    #itemsToRemove = set(["HalfBath"])  # changed to ScreenPorch from ScreenProch
    # TODO Why does the result different from the result in the while loop above? This should match up with the accuracy 0.97021332160605. Does the order of list matter??

    #post_featureRemoval = filter(lambda x: x not in itemsToRemove, inputsCol)
    # post_featureRemoval = list(set(inputsCol) - itemsToRemove)
    # inputsCol.remove('HalfBath')
    #alg = GradientBoostingRegressor(random_state = 1)
    #cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, post_featureRemoval], trainDF.loc[:, outputCol], cv=10, scoring='r2')
    #print("After removing all detrimental features =", np.mean(cvScores))
    # This results in a lower value... does this mean some of these are related, or simply need preprocessing
    #   (e.g., year built should probably become age)?

    #visualization.visualize(trainDF,inputsCol,outputCol)
    
    #Testing a different model (linear regression)
    """
    alg = LinearRegression()
    cvScores = model_selection.cross_val_score(alg, trainDF.loc[:, inputsCol], trainDF.loc[:, outputCol], cv=10, scoring='mean_squared_error')
    print("Accuracy for linear regression:", np.mean(cvScores))
    """
    # This code is suggested to be correct, even though it returns the very high MSE of -3.902505273678058e+32
    # Supported by https://stackoverflow.com/questions/24132237/scikit-learn-cross-validation-scoring-for-regression

main()