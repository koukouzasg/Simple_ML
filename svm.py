import numpy as np
import utils
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.svm import LinearSVC
# fix random seed reproducibility
np.random.seed(7)

print("Type 1 for linear Kernel else Gaussian will be used")
type = int(input())

dataset = np.loadtxt("spambase.txt", delimiter=',')
total_accuracy = 0
total_f1_score = 0

kfold = KFold(10, True, 1)

for train, test in kfold.split(dataset):
    data, target = utils.prepare_data_mlp(dataset, train)
    test_data, expected = utils.prepare_data_mlp(dataset, test)

    if type == 1:
        model = LinearSVC()
    else:
        model = svm.SVC(kernel='rbf')

    model.fit(data, target)

    predicted = model.predict(test_data)

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    f1_score = metrics.f1_score(expected, predicted)
    accuracy_score = metrics.accuracy_score(expected, predicted)
    print("The accuracy is : {}\nThe f1 score is : {}\n".format(accuracy_score, f1_score))

    total_accuracy = total_accuracy + accuracy_score
    total_f1_score = total_f1_score + f1_score

utils.calculate_results(total_accuracy, total_f1_score)
