from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from utils import load_datasets
from model import Dataset
from utils import FeatureSelector
from model import Patient
from utils import SplitSets

from algorithms import KNearestNeighbour

import os
import csv
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.svm import SVR



def main():
    path_to_dataset = "C:\\GIT\\ZIWM\\data.csv"
    total_number_of_classes = 8
    total_number_of_features = 30

    #loading raw values from file
    datasetLoader = load_datasets.Load_datasets(path_to_dataset)
    dataset_raw_values = datasetLoader.loadDataset()

    #constructing main dataset with division for features and classes
    dataset = Dataset(dataset_raw_values)

    # selecting number of features and running tests
    for number_of_features_selected in range(1,25):
        print(number_of_features_selected)
        trimmed_feature_list = FeatureSelector.selectKBestFeatures(number_of_features_selected,
                                                                   dataset.dataset_features_array,
                                                                   dataset.dataset_class_array)
        #   dividing data sets into patients
        patients=[]
        for i in range(len(dataset.dataset_class_array)):
            patient = Patient(i, dataset.dataset_class_array[i], trimmed_feature_list[i])
            print(patient.getId(), patient.getDisease_class(), patient.get_features())
            patients.append(patient)


        #   creating learn and test data sets
        learning_set, testing_set = SplitSets.splitSets(patients)

        kn = KNearestNeighbour(5,"minkowski", learning_set, testing_set)

        kn.train()

        #   creating algorythm and training
        #   comparing results of test data set
        #   calculating hit rate


main()



# data = load_iris()
# print(data)
# X, y = load_iris(return_X_y=True)
# print(X.shape)
# print(X)
# print("//////////////////////////////\n\////////////////////////")
# print(y)
# X_new = SelectKBest(chi2, k=2).fit_transform(X,y)
# print("//////////////////////////////\n\////////////////////////")
# print(X_new.shape)
# print(X_new)

#
# script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
#
# rel_path = "../data.cvs"
# #abs_file_path = os.path.join(script_dir, rel_path)
# abs_file_path = "C:\\GIT\\ZIWM\\data.csv"
#
# print(rel_path)
# print(script_dir)
# print(abs_file_path)
#
#
# csv_context = []
# with open(abs_file_path) as csvfile:
#     reader = csv.reader(csvfile, delimiter=";") # change contents to floats
#     for row in reader: # each row is a list
#         csv_context.append(row)
#
#
# print(csv_context)
#
# features_array = ([])
# class_assignment_array = np.array([])
#
#
#
# for line in csv_context:
#
#     print("\n")
#     temp_arr = ([])
#
#     for column in range(len(line)):
#         if column == len(line)-1:
#             class_assignment_array = np.append(class_assignment_array, line[column])
#         else:
#
#             temp_arr.append(line[column])
#
#     print("xxx xxxx")
#     print(temp_arr)
#     print(features_array)
#     features_array.append(temp_arr)
#     print(features_array)
#
#
#
#
# print("\nfeatures array ")
# print(features_array)
# print(len(features_array))
# print("class_assignemnt ")
# print(class_assignment_array)
# print(len(class_assignment_array))
#
#
# features_array = np.array(features_array)
# features_array = features_array.astype(float)
# class_assignment_array = class_assignment_array.astype(float)
#
#
# NUM_OF_FEATURES = 25
#
# selectorCHI = SelectKBest(chi2, k=NUM_OF_FEATURES)
#
# selected_features = selectorCHI.fit_transform(features_array.tolist(), class_assignment_array.tolist())
# print(selected_features)
# print("CHI:")
# print(selectorCHI.get_support())
#
#
# estimator = SVR(kernel="linear")
# selectorRFE = RFE(estimator, NUM_OF_FEATURES, step=1)
# selectorRFE = selectorRFE.fit(features_array, class_assignment_array)
# print("SVR:")
# print(selectorRFE.support_)
# print("SVR ranking: ")
# print(selectorRFE.ranking_)
#
#
# print("feature selected by both metods")
# for x in range(len(selectorRFE.support_)):
#     print(x+1)
#     print(selectorRFE.support_[x] & selectorCHI.get_support()[x])
#





