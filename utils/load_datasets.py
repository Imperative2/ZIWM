import os
import csv
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.svm import SVR


class Load_datasets:

    path_to_file = ""

    def __init__(self, path_to_file="asdf"):
        self.path_to_file = path_to_file
        print("class initialized")


    def loadDataset(self):
        csv_context = []
        with open(self.path_to_file) as csvfile:
            reader = csv.reader(csvfile, delimiter=";")
            for row in reader:
                csv_context.append(row)
        return csv_context

