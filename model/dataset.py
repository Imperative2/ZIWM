class Dataset:
    dataset_raw_values = []
    dataset_class_array = []
    dataset_features_array = []
    def __init__(self,dataset_raw_values):
        self.dataset_raw_values = dataset_raw_values
        self.dataset_class_array = self.getClassesFromDataset(self.dataset_raw_values)
        self.dataset_features_array = self.getFeaturesFromDataset(self.dataset_raw_values)
        print("classes")
        print(self.dataset_class_array)
        print("features")
        print(self.dataset_features_array)



    def getClassesFromDataset(self, raw_values):
        classes_array = ([])
        for row in raw_values:
            classes_array.append(row[len(row)-1])
        return classes_array

    def getFeaturesFromDataset(self, raw_values):
        features_array = ([])
        for row in raw_values:
            temp_arr = row[0:(len(row)-1)]
            features_array.append(temp_arr)
        return features_array

