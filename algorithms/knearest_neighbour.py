from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNearestNeighbour:

    number_of_neighbours = 5
    metric = "minkowski"
    test_set = []
    training_set = []
    classifier = None

    def __init__(self, number_of_neighbours, metric, training_set, test_set):
        self.number_of_neighbours = number_of_neighbours
        self.metric = metric
        self.training_set = training_set
        self.test_set = test_set

        self.classifier = KNeighborsClassifier(n_neighbors=number_of_neighbours)

    def setTestSet(self, test_set):
        self.test_set = test_set

    def setTrainingSet(self, training_set):
        self.training_set = training_set

    def train(self):
        training_features = ([])
        training_classes = []
        for patient in self.training_set:
         #  print(patient)
            training_features.append(patient.get_features())
            training_classes.append(patient.getDisease_class())
      #  print("training_features")
      #  print(training_features)
       # print("training_classes")
       # print(training_classes)

        self.classifier.fit(training_features, training_classes)

    def test(self):
        good_fit = len(self.test_set)

        for patient in self.test_set:
            temp = np.array(patient.get_features())
            temp2 = []
          #  print("patient features length")
           # print(len(patient.get_features()))
            temp.reshape((1,-1))
           # print(temp)
           # print(patient.get_features())
            temp2.append(patient.get_features())
           # print(temp2)
            prediction = self.classifier.predict(temp2)
            # if len(patient.get_features() == 1):
            #     prediction = self.classifier.predict(patient.get_features().reshape(-1,1))
            # else:
            #     prediction = self.classifier.predict(patient.get_features())
        #    print("prediction and reality:")
        #    print(prediction, patient.getDisease_class())

            if prediction[0] != patient.getDisease_class():
          #      print(prediction[0])
                good_fit-=1
          #  else:
            #    print("bad prediction")

        return good_fit/len(self.test_set)
