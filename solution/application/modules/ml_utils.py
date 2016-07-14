__author__ = 'marko'

from sklearn.learning_curve import learning_curve

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import  GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,precision_recall_curve, mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, train_sizes=np.linspace(.3, 1.0, 10)):
    """
    Plot learning curve for given model
    :param estimator:
    :param title:
    :param X:
    :param y:
    :param ylim:
    :param cv:
    :param n_jobs:
    :param train_sizes:
    :return:
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



def printScore(y_true,y_pred,average='micro',pos_label=None):
    """
    Print model scores
    :param y_true:
    :param y_pred:
    :param average:
    :param pos_label:
    :return:
    """
    print average+" scores:"
    print "\t P  = %s" % precision_score(y_true,y_pred,average=average,pos_label=pos_label)
    print "\t R  = %s" % recall_score(y_true,y_pred,average=average,pos_label=pos_label)
    print "\t F1 = %s" % f1_score(y_true,y_pred,average=average,pos_label=pos_label)



def _model_evaluator(X, y, model, parameters, scoring = None, num_folds = 3,test_size = 0.3,
                    ylim=None,train_sizes_lncurv=np.linspace(.1, 1.0, 10), verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    grid_cv = GridSearchCV( model, parameters, scoring = scoring, n_jobs = -1, verbose = 1, cv = num_folds)
    grid_cv.fit(X_train,y_train)

    estimator = grid_cv.best_estimator_
    if verbose:
        print 'Model %s' % estimator
        print 'Model best_params: %s' % grid_cv.best_params_
        print 'Model score : %s' % estimator.score(X_test,y_test)

    y_pred = estimator.predict(X_test)
    if not verbose: return estimator

    print "Confusion matrix:\n %s" % confusion_matrix(y_test,y_pred)
    if len(set(y)) == 2:
        printScore(y_test,y_pred,'binary',list(set(y))[0])
    else:
        printScore(y_test,y_pred,'macro')
        printScore(y_test,y_pred)
    #plot_learning_curve(estimator, "Learning curve " + str(model).split('(')[0], X, y,  ylim=ylim, cv=num_folds,train_sizes=train_sizes_lncurv)

    return estimator

def evaluate_models(X, y, models, parameters, num_folds=10, k=20, verbose=False):
    sel = SelectKBest(f_classif, k=k)
    X_reduced = sel.fit_transform(X, y)

    for model, parameter in zip(models,parameters):
        _model_evaluator(X_reduced, y, model, parameter, num_folds=num_folds, verbose=verbose)


def store_model(path, model):
    """
    Method used for persisting trained model to file.
    :param path:
    :param model:
    :return:
    """
    _ = joblib.dump(model, path, compress=9)

def load_model(path):
    """
    Load trained model from file
    :param path:
    :return:
    """
    return joblib.load(path)

CLASSIFIERS = [DummyClassifier(), GaussianNB(), LinearSVC(), SVC(), DecisionTreeClassifier(), RandomForestClassifier()]
PARAMETERS = [{}, {}, {},{'kernel':('linear', 'rbf', 'poly', 'sigmoid')},{}, {}]
