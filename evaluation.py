__author__ = 'fabien'

import numpy as np
from matplotlib import pyplot as plt
from sklearn.learning_curve import validation_curve
import seaborn as sns


def deviance_curve(classifier, features, labels, metaparameter_name, param_range, metric='Accuracy',
                   n_folds=4, njobs=-1, fig_size=(16, 9)):

    training_scores, validation_scores = validation_curve(classifier,
                                                      features, labels,
                                                      metaparameter_name,
                                                      param_range,
                                                      n_jobs=njobs,
                                                      cv=n_folds, scoring=metric)

    training_scores_mean = np.mean(training_scores, axis=1)
    training_scores_std = np.std(training_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    plt.figure(num=None, figsize=fig_size, dpi=600, facecolor='w', edgecolor='k')
    plt.title("Validation Curve")
    plt.xlabel(metaparameter_name)
    plt.ylabel(metric)
    plt.xlim(np.min(param_range), np.max(param_range))
    plt.semilogx(param_range, training_scores_mean, label="Training " + metric, color="mediumblue")
    plt.fill_between(param_range, training_scores_mean - training_scores_std,
                     training_scores_mean + training_scores_std, alpha=0.2, color="lightskyblue")
    plt.semilogx(param_range, validation_scores_mean, label="validation " + metric,
                 color="coral")
    plt.fill_between(param_range, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.2, color="lightcoral")
    plt.legend(loc="best")
    plt.show()
