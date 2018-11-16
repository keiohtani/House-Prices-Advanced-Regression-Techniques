import pandas as pd

def readData(numRows = None):
    # inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    # outputCol = 'Class'
    # colNames = [outputCol] + inputCols  # concatenate two lists into one
    # wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)
    # wineDF = wineDF.sample(frac=1,random_state=99).reset_index(drop=True)
    #
    trainDF = pd.read_csv("data/train.csv")
    return trainDF

def main():
    trainDF = readData()
    print(trainDF.loc[0, :])

main()