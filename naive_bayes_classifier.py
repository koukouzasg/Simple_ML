from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import utils

df = utils.load_spam_csv()

total_accuracy = 0
total_f1_score = 0

kfold = KFold(10, True, 1)

for train, test in kfold.split(df):
    # Set the correct format for the data to be fed into the model
    data, target = utils.prepare_data_bayes(df, train)
    test_data, expected = utils.prepare_data_bayes(df, test)

    # Train the model
    model = GaussianNB()
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
