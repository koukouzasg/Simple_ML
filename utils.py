import pandas as pd
import numpy as np

def calculate_results(total_accuracy, total_f1_score):
    total_accuracy = total_accuracy / 10
    total_f1_score = total_f1_score / 10

    print("Total accuracy : {}".format(total_accuracy))
    print("Total f1 score : {}".format(total_f1_score))


def load_spam_csv():
    df = pd.read_csv('spambase.csv', ';', header=None)
    print(df.head())
    return df

def prepare_data_bayes(df, subset):
    data = []
    target = []
    for index in subset:
        row = df.iloc[[index]].values.tolist()
        row = row[-1]
        target.append(row[-1])
        data.append(row[:-1])
    return data, target

def prepare_data_mlp(dataset, subset):
    data = []
    target = []
    for index in subset:
        row = dataset[index]
        target.append(row[-1])
        data.append(row[:-1])
    data = np.array(data)
    target = np.array(target)
    return data, target

def get_inputs():
    print('Enter the number of k1 hidden neurons:')
    k1 = int(input())
    print('Enter the number of k2 hidden neurons:')
    k2 = int(input())
    return k1, k2





