from sklearn.neighbors import KNeighborsClassifier


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

        self.classifier = KNeighborsClassifier

    def setTestSet(self, test_set):
        self.test_set = test_set

    def setTrainingSet(self, training_set):
        self.training_set = training_set

    def train(self):
        for patient in self.training_set:
            print(patient)
