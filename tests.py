from sklearn import discriminant_analysis as da
import pickle



def frozen(*arg):
    raise AttributeError("This method has been removed")


def svm_analysis(data_set):
    events_features, categories = None, None
    analyser = da.LinearDiscriminantAnalysis()
    # analyser.fit(events_features, categories)
    file = open('figs/svm_classifier', mode='wb')
    # print('Classification score: ' + str(analyser.score(events_features, categories)))
    analyser.fit = frozen
    pickle.dump(analyser, file)

svm_analysis('a')
file = open('figs/svm_classifier', mode='rb')
u = pickle.load(file)
u.fit()