import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt

def trainData(df, clf):
    result = list(df.revenue)
    trainResult = result[:2990]
    clf.fit(df[['budget', 'popularity', 'runtime']].head(2990), trainResult)
    savFile = 'trainedModel.sav'
    joblib.dump(clf, savFile)

# testdf = pd.read_csv('test.csv')
# testdf.fillna(testdf.mean())
# testres = list(testdf.revenue)

def predict(clf, df):
    pr = clf.predict(df[['budget', 'popularity', 'runtime']].tail(10))
    return list(pr)

def listToDict(li):
    result = {}
    for x in range(len(li)):
        result[x] = li[x]
    return result

def plotActualPredicted(actdata, predata, names):
    sns.set()
    percentile_list = pd.DataFrame(
        {
            u'actual': listToDict(actdata),
            u'predicted': listToDict(predata),
            u'movies': listToDict(names)
        }
    )   
    percentile_list = percentile_list.set_index('movies')
    fig = plt.figure(figsize=(10,10)) # Create matplotlib figure

    ax = fig.add_subplot(111) # Create matplotlib axes
    ax2 = ax.twinx() # Create another axes that shares the same x-axis as a
    width = .3

    percentile_list.predicted.plot(kind='bar',color='green',ax=ax,width=width, position=0)
    percentile_list.actual.plot(kind='bar',color='blue', ax=ax2,width = width,position=1)

    ax.grid(None)
    ax2.grid(None)

    ax.set_ylabel('actual revenue (blue)')
    ax2.set_ylabel('predicted revenue (green)')

    ax.set_xlim(-1,10)
    # plt.autoscale(enable=True, axis='y')
    # plt.yticks(np.arange(0, 1000000000, 100))
    plt.show()

def predictMovie(movieEx, clf):
    movie = pd.DataFrame(
        {
            u'budget': {0: movieEx['budget']},
            u'popularity': {0: movieEx['popularity']},
            u'runtime': {0: movieEx['runtime']}
        }
    )  
    return list(clf.predict(movie[['budget', 'popularity', 'runtime']]))[0]

def main():
    clf = RandomForestClassifier(n_jobs=2, random_state=0, warm_start=False)
    filename = 'train.csv'
    resdf = pd.read_csv(filename)
    resdf = resdf.fillna(resdf.mean())

    # Save the model
    trainData(resdf, clf)

    # Load the model
    # loadFile = 'trainedModel.sav'
    # loaded_clf = joblib.load(loadFile)
    # Specific movie example
    movieExample = {'budget': 100000000, 'popularity':7.94, 'runtime': 94}
    print(predictMovie(movieExample, clf))

    actResult = list(resdf.revenue)[2990:3000]
    preresult = predict(clf, resdf)
    plotActualPredicted(actResult, preresult, list(resdf['title'].tail(10)))
    print(actResult)
    print(preresult)

main()
