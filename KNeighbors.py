from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import utils
import numpy as np


print('Enter the selected number of neighbors:')
k = int(input())

dataset = np.loadtxt("spambase.txt", delimiter=',')
total_accuracy = 0
total_f1_score = 0

kfold = KFold(10, True, 1)

for train, test in kfold.split(dataset):
    data, target = utils.prepare_data_mlp(dataset, train)
    test_data, expected = utils.prepare_data_mlp(dataset, test)

    neigh = KNeighborsClassifier(n_neighbors= k , weights= 'uniform')
    neigh.fit(data, target)

    predicted = neigh.predict(test_data)

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    f1_score = metrics.f1_score(expected, predicted)
    accuracy_score = metrics.accuracy_score(expected, predicted)
    print("The accuracy is : {}\nThe f1 score is : {}\n".format(accuracy_score, f1_score))

    total_accuracy = total_accuracy + accuracy_score
    total_f1_score = total_f1_score + f1_score

utils.calculate_results(total_accuracy, total_f1_score)
