import seaborn as sns
import matplotlib.pyplot as plt

def visualize(df, inputsCol, outputCol):
    print('Visualization begins')
    for col in inputsCol:
        sns.jointplot(outputCol, col, df)
        # sns.lmplot('SalePrice', col, df)  # draws a line in the graph, I did not see the use
        # sns.catplot(x=outputCol, y=col, data=df)  # takes too much time to run
        # sns.swarmplot(x=outputCol, y=col, data=df)    # similar to jointplot but it spreads the points depending on the number of items for each value
        # sns.scatterplot(x=outputCol, y=col, data=df)    # scatterplot does not require values to be numerical so it is good for analysis
    plt.show()
    print("visualization finished")

def visualizeScatterplot(df, inputsCol, outputCol):
    print('Visualization begins')
    for col in inputsCol:
        # sns.jointplot('SalePrice', col, df)
        # sns.lmplot('SalePrice', col, df)  # draws a line in the graph, I did not see the use
        # sns.catplot(x=outputCol, y=col, data=df)  # takes too much time to run
        # sns.swarmplot(x=outputCol, y=col, data=df)    # similar to jointplot but it spreads the points depending on the number of items for each value
        sns.scatterplot(x=outputCol, y=col, data=df)    # scatterplot does not require values to be numerical so it is good for analysis
        plt.show()
    print("visualization finished")
