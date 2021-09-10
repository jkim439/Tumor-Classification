import os
from sklearn.model_selection import KFold
import pandas as pd

from random import shuffle

path = "/home/jkim/Project/references/classification/data/"

list_ADIMUC = os.listdir(path + "ADIMUC/")
list_STRMUS = os.listdir(path + "STRMUS/")
list_TUMSTU = os.listdir(path + "TUMSTU/")

list = list_ADIMUC + list_STRMUS + list_TUMSTU
shuffle(list)

kfold = KFold(n_splits=5, shuffle=True)

k = 1
for train_index, test_index in kfold.split(list):

    path_new = path + str(k) + "/"
    os.makedirs(path_new, exist_ok=True)

    ##### TRAIN #####
    train_file = []
    train_class = []
    for i in range(len(train_index)):
        if list[train_index[i]][:3] == "ADI" or list[train_index[i]][:3] == "MUC":
            train_file.append(str(path + "ADIMUC/") + str(list[train_index[i]]))
            train_class.append(int(0))
        elif list[train_index[i]][:3] == "STR" or list[train_index[i]][:3] == "MUS":
            train_file.append(str(path + "STRMUS/") + str(list[train_index[i]]))
            train_class.append(int(1))
        elif list[train_index[i]][:3] == "TUM" or list[train_index[i]][:3] == "STU":
            train_file.append(str(path + "TUMSTU/") + str(list[train_index[i]]))
            train_class.append(int(2))
        else:
            print("ERROR: Failed to get class.")
            exit(1)
    df_train = pd.DataFrame(train_file, columns=["filename"])
    df_train["class"] = train_class
    df_train.to_csv(path_new + "train.csv", columns=["filename", "class"], index=False, header=False)

    ##### TEST #####
    test_file = []
    test_class = []
    for j in range(len(test_index)):
        if list[test_index[j]][:3] == "ADI" or list[test_index[j]][:3] == "MUC":
            test_file.append(str(path + "ADIMUC/") + str(list[test_index[j]]))
            test_class.append(int(0))
        elif list[test_index[j]][:3] == "STR" or list[test_index[j]][:3] == "MUS":
            test_file.append(str(path + "STRMUS/") + str(list[test_index[j]]))
            test_class.append(int(1))
        elif list[test_index[j]][:3] == "TUM" or list[test_index[j]][:3] == "STU":
            test_file.append(str(path + "TUMSTU/") + str(list[test_index[j]]))
            test_class.append(int(2))
        else:
            print("ERROR: Failed to get class.")
            exit(1)
    df_test = pd.DataFrame(test_file, columns=["filename"])
    df_test["class"] = test_class
    df_test.to_csv(path_new + "test.csv", columns=["filename", "class"], index=False, header=False)

    total = len(df_train.index) + len(df_test.index)

    if total != 11977:
        print("ERROR: Failed to verify.")
        exit(1)

    print(str(k) + ". TOTAL: " + str(total) + " / Train: " + str(len(df_train.index)) + " / Test: " + str(
        len(df_test.index)))

    k += 1
