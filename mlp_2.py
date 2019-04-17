from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import utils
from sklearn import metrics
# fix random seed reproducibility
from sklearn.model_selection import KFold

np.random.seed(7)

k1, k2 = utils.get_inputs()

dataset = np.loadtxt("spambase.txt", delimiter=',')
total_accuracy = 0
total_f1_score = 0
kfold = KFold(10, True, 1)

for train, test in kfold.split(dataset):
    data, target = utils.prepare_data_mlp(dataset, train)
    test_data, expected = utils.prepare_data_mlp(dataset, test)

    # Create Model
    model = Sequential()
    model.add(Dense(12, input_dim=57, activation='sigmoid'))
    model.add(Dense(k1, activation='sigmoid'))
    model.add(Dense(k2, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
    # Fit the model
    model.fit(data, target, epochs=150, batch_size=32)
    # Predict results
    predicted = model.predict(test_data)
    # Round predictions
    predicted_rounded = [round(x[0]) for x in predicted]

    print(metrics.classification_report(expected, predicted_rounded))
    print(metrics.confusion_matrix(expected, predicted_rounded))

    f1_score = metrics.f1_score(expected, predicted_rounded)
    accuracy_score = metrics.accuracy_score(expected, predicted_rounded)
    print("The accuracy is : {}\nThe f1 score is : {}\n".format(accuracy_score, f1_score))

    total_accuracy = total_accuracy + accuracy_score
    total_f1_score = total_f1_score + f1_score

utils.calculate_results(total_accuracy, total_f1_score)