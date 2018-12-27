""" _________________________ Importing Module(s) __________________________"""


import numpy as np
import matplotlib.pyplot as plt


'''_________________________ Confusion Matrix ______________________________'''


def confusion_matrix(
        yes_examples, no_examples, network, features_test, labels_test, ds_id):

    '''
        Args:
            yes_examples: 1D logical array with examples of class + equal to 1;
            yes_examples: 1D logical array with examples of class - equal to 1;
            network: Object of class Network;
            features_test: DataFrame containing test examples features;
            labels_test: DataFrame containing test examples labels;
            ds_id: Dataset ID: 0 :: Banknote | 1 :: Diabetes
    '''

    num_pos = np.sum(yes_examples)  # the number of real positive cases
    num_neg = np.sum(no_examples)  # the number of real negative cases
    # OPT: print("+: ",num_pos,"-: ",num_neg)
    threshold = np.linspace(1e-15, 1-1e-15, 100)
    pred_target = []
    tpr = []  # Recall
    tnr = []
    fpr = []
    fnr = []
    precision_ = []
    recall = []
    accuracy = []
    for th in threshold:
        pred_target_, tp_, tn_, fp_, fn_ = network.test(
                features_test, labels_test, th)
        pred_target.append(pred_target_)
        tpr.append(tp_/num_pos)
        tnr.append(tn_/num_neg)
        fpr.append(fp_/num_neg)
        fnr.append(fn_/num_pos)
        if (tp_+fp_) != 0:
            precision_.append(tp_/(tp_+fp_))
            recall.append(tp_/num_pos)
        accuracy.append(100*(tp_+tn_)/(num_pos+num_neg))

    # OPT: print("TPR: ",tpr)

    if not ds_id:
        title_ = 'ROC curve for the Banknote Classifier'
    else:
        title_ = 'ROC curve for the Diabetes Classifier'

    _plot_eval_curves(2, fpr, tpr, threshold, threshold, [0.0, 1.0],
                      [0.0, 1.0], 'False Positive Rate (1 - Specificity)',
                      'True Positive Rate (Sensitivity)', title_)

    precision = []
    recall.reverse()
    precision_.reverse()
    for i in range(len(recall)):
        precision.append(max(precision_[i:]))  # interpolated precision

    ydash_ = (num_pos/(num_pos+num_neg))*np.ones([len(threshold), 1])
    if not ds_id:
        title_ = 'PR curve for the Banknote Classifier'
    else:
        title_ = 'PR curve for the Diabetes Classifier'
    _plot_eval_curves(3, recall, precision, threshold, ydash_, [0.0, 1.0],
                      [0.0, 1.0], 'Recall (True Positive Rate)', 'Precision',
                      title_)

    if not ds_id:
        title_ = 'Accuracy for the Banknote Classifier'
    else:
        title_ = 'Accuracy for the Diabetes Classifier'

    _plot_eval_curves(4, threshold, accuracy, None, None, [0.0, 1.0],
                      [0.0, 100.0], 'Threshold', 'Accuracy', title_)

    exit_msg = 'Success!\n'
    return exit_msg


def _plot_eval_curves(plot_num, x, y, xdash, ydash, xlim_, ylim_, xlabel_,
                      ylabel_, title_):
    plt.figure(plot_num)
    plt.plot(x, y, linewidth=5.0)
    if xdash is not None and ydash is not None:
        plt.plot(xdash, ydash, linestyle='dashed')
        plt.axis('square')
    plt.xlim(xlim_)
    plt.ylim(ylim_)
    plt.rcParams['font.size'] = 12
    plt.title(title_)
    plt.xlabel(xlabel_)
    plt.ylabel(ylabel_)
    plt.grid(True)
