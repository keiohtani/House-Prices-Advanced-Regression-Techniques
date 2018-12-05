from sklearn import preprocessing

"""
Various conversion functions

Parameters:
    x = data value provided by caller algorithm (value to be converted)

Return:
    Returns a constant value based on the function and the formal parameter

Description:
    Converts nominal data into standardized numeric data
"""


def masVnrTypeConversion(x):  # https://www.angieslist.com/articles/how-much-does-brick-veneer-cost.htm
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


def poolQCConversion(x):  # there is no poor for pool
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
    
def dateToAgeConversion(x): #Converting data for which the date is supplied to the appropriate age in years
    return 2018 - x


"""
normalization and standardization

Parameters:
    - targetDF, df = the DataFrame on which the function operates
    - columns = the columns in which data values are to be normalized/standardized

Return:
    None

Description:
    Normalizes/standardizes numeric data in a DataFrame provided by the caller function
"""


def normalization(targetDF, columns):
    targetDF.loc[:, columns] = (targetDF.loc[:, columns] - targetDF.loc[:, columns].min()) / (
                targetDF.loc[:, columns].max() - targetDF.loc[:, columns].min())


def standardize(df, columns):
    df.loc[:, columns] = (df.loc[:, columns] - df.loc[:, columns].mean()) / df.loc[:, columns].std()


"""
encodeNominalData

Parameters:
    - inputDF = the DataFrame on which the function operates
    - inputCols = the columns in which data are to be encoded

Return:
    None

Description:
    Blindly converts nominal data into numeric data (e.g., first data point = 0, second = 1, and so on)

Original code, based on https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
"""
#"""
def encodeNominalData(inputDF, inputCols):
    reps = len(inputCols)
    for i in range(reps):
        # print(inputCols[i])
        labelEncoder = preprocessing.LabelEncoder()
        labelEncoder.fit(inputDF.loc[:,inputCols[i]])
        inputDF.loc[:,inputCols[i]] = labelEncoder.transform(inputDF.loc[:,inputCols[i]])
#"""

"""
manageNAValues

Parameters:
    - targetDF = the DataFrame on which the function operates
    - columns = the columns in which data values are to be converted

Return:
    None

Description:
    Fills all "NA" values in the DataFrame:
        If "NA" is significant, it becomes "NoValue" and is then processed by the encoder
        If "NA" is not significant, it becomes the mode of elements in the DataFrame
        If "NA" means something doesn't exist, it converts to a value of 0.

        TODO:Make sure all 0 values are accounted for
"""


def manageNAValues(inputDF, inputCols):
    inputDF.loc[:, 'LotFrontage'] = inputDF.loc[:, 'LotFrontage'].fillna(0)
    # First, deal with columns where "NA" actually means something
    significantNAFields = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
                           "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
                           "PoolQC", "Fence", "MiscFeature"]
    for columnName in significantNAFields:
        inputDF.loc[:, columnName].map(lambda val: "NoValue" if val == "NA" else val)

    # Then, fill in remaining "NA" values (currently with mode of other column entries)
    for column in inputCols:
        inputDF.loc[:, column] = inputDF.loc[:, column].fillna(inputDF.loc[:, column].mode()[0])


"""
preprocess

Parameters:
    - targetDF = the DataFrame on which the function operates
    - sourceDF = the training DataFrame
    - inputsCol = the columns in which data values are to be edited

Return:
    None

Description:
    An aggregate function for all preprocessing function calls (to avoid multiple calls in main)
"""


def preprocess(targetDF, sourceDF, inputsCol):
    manageNAValues(targetDF, inputsCol)
    outputCol = 'SalePrice'
    exConversionCols = ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
    """
    - operation of lists
    https://stackoverflow.com/questions/3428536/python-list-subtraction-operation 
    """

    # outputSeries = (trainDF.loc[:, outputCol] - trainDF.loc[:, outputCol].min()) / (trainDF.loc[:, outputCol].max() - trainDF.loc[:, outputCol].min())
    # convertNominalValue(trainDF, outputSeries, inputsCol)
    
    nominalDataCol = list(set(inputsCol) - set(exConversionCols) - set(
        ['BsmtQual', 'PoolQC', 'MasVnrType', 'Fence', '1stFlrSF', 'GarageArea']))  # - set(numericDataCols)

    convertNominalValue(targetDF, sourceDF, nominalDataCol, outputCol)
    #Accuracy with ONLY convertNominalValue applied, including normalization = 0.9684257179045664
    
    #encodeNominalData(targetDF, inputsCol) #yields 0.8879956626324738
    targetDF.loc[:, exConversionCols] = targetDF.loc[:, exConversionCols].applymap(
        lambda x: nominalValueConversion(x))  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    # targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]] = targetDF.loc[:, ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]].applymap(lambda x: 1.0 if x == 'Ex' else (0.75 if x == 'Gd' else (0.5 if x == 'TA' else (0.25 if x == 'Fa' else (0.0 if x == 'Po' else x)))))    # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].map(lambda x: bsmtQualConversion(x))
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].map(lambda x: poolQCConversion(x))
    targetDF.loc[:, 'MasVnrType'] = targetDF.loc[:, 'MasVnrType'].map(lambda x: masVnrTypeConversion(x))
    targetDF.loc[:, 'Fence'] = targetDF.loc[:, 'Fence'].map(lambda x: fenceValueConversion(x))
    # targetDF.loc[:, "LotShape"] = targetDF.loc[:, "LotShape"].map(lambda x: lotShapeValueConversion(x))
    targetDF.loc[:, 'YearBuilt'] = targetDF.loc[:, 'YearBuilt'].map(lambda x: dateToAgeConversion(x)) #went from 0.9695631660091774 to 0.9695838362322334
    #targetDF.loc[:, 'YrSold'] = targetDF.loc[:, 'YrSold'].map(lambda x: dateToAgeConversion(x)) #Results in accuracy decrease
    # standardize(targetDF, inputsCol)  # Accuracy 0.8952159968525466
    normalization(targetDF, inputsCol)  # Accuracy 0.8958124722966672
    # targetDF.loc[:, "YearRemodAdd"] = targetDF.loc[:, ['YearBuilt', "YearRemodAdd"]].apply(lambda row: np.NaN if row.loc['YearBuilt'] == row.loc['YearRemodAdd'] else row.loc['YearRemodAdd'], axis = 1) # remodel year should be adjusted in the case of remodel has not been done."""
'''
def testPreprocess(targetDF, sourceDF, inputsCol, col):     # test function to see the difference between different value to be encoded.
    exConversionCols = ["ExterQual", "ExterCond", "BsmtCond", "KitchenQual", "FireplaceQu", "GarageQual",
                        "GarageCond"]
    """
    - operation of lists
    https://stackoverflow.com/questions/3428536/python-list-subtraction-operation
    """
    print(col)
    nominalDataCol = list(set(inputsCol) - set(exConversionCols) - set(['BsmtQual', 'PoolQC', 'MasVnrType', 'Fence', '1stFlrSF', 'GarageArea', col]))  # - set(numericDataCols)
    # encodeNominalData(targetDF, nominalDataCol)
    targetDF.loc[:, exConversionCols] = targetDF.loc[:, exConversionCols].applymap(lambda x: nominalValueConversion(x))  # Ex, Gd, TA, Fa, Po to be numerical value 1.0, 0.75, 0.5, 0.25 0.0
    targetDF.loc[:, "BsmtQual"] = targetDF.loc[:, "BsmtQual"].map(lambda x: bsmtQualConversion(x))
    targetDF.loc[:, "PoolQC"] = targetDF.loc[:, "PoolQC"].map(lambda x: poolQCConversion(x))
    targetDF.loc[:, 'MasVnrType'] = targetDF.loc[:, 'MasVnrType'].map(lambda x: masVnrTypeConversion(x))
    targetDF.loc[:, 'Fence'] = targetDF.loc[:, 'Fence'].map(lambda x: fenceValueConversion(x))
    # targetDF.loc[:, "LotShape"] = targetDF.loc[:, "LotShape"].map(lambda x: lotShapeValueConversion(x))
    # standardize(targetDF, inputsCol)  # Accuracy 0.8952159968525466
    normalization(targetDF, inputsCol)  # Accuracy 0.8958124722966672
    # targetDF.loc[:, "YearRemodAdd"] = targetDF.loc[:, ['YearBuilt', "YearRemodAdd"]].apply(lambda row: np.NaN if row.loc['YearBuilt'] == row.loc['YearRemodAdd'] else row.loc['YearRemodAdd'], axis = 1) # remodel year should be adjusted in the case of remodel has not been done.
'''

"""
convertNominalValue

Parameters:
    - targetDF = the DataFrame on which the function operates
    - sourceDF = the training DataFrame
    - inputCols = the columns in which data values are to be edited
    - outputCol = the class variable column designation

Return:
    None

Description:
    Encodes data values based on the median price of all houses with that value
"""


def convertNominalValue(targetDF, sourceDF, inputCols, outputCol):
    print('Conversion begins')
    for colName in inputCols:
        # colName = 'MSSubClass'  # each column fx. MSSubClass
        aveDF = sourceDF.loc[:, [colName, outputCol]].groupby(
            [colName]).median()  # returns DataFrame somehow so needs to be converted to Series
        # TODO: though accuracy is better with median, it may need to be adjusted depending on the distribution
        aveSeries = aveDF.iloc[:, 0]
        colSeries = targetDF.loc[:, colName]
        targetDF.loc[:, colName] = colSeries.index.map(lambda i: aveSeries.loc[colSeries.iloc[i]])
        # for col in inputCols:
        #     # col = 'MSSubClass'  # each column fx. MSSubClass
        #     aveSeries = trainDF.loc[:, [col, outputCol]].groupby([col]).median()
        #     colSeries = trainDF.loc[:, col]
        #     trainDF.loc[:, col] = colSeries.index.map(lambda i: aveSeries.loc[colSeries.iloc[i]])
    print('Conversion ended')
