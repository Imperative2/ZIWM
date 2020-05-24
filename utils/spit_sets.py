import random


class SplitSets:

    @staticmethod
    def splitSets(patients):
        learning_set = []
        test_set = []
        half_of_patients = int(len(patients)/2)
        drawn_numbers = []

        while len(learning_set) < half_of_patients:
            rand_number = random.randint(0, len(patients)-1)
            if rand_number not in drawn_numbers:
                learning_set.append(patients[rand_number])
                drawn_numbers.append(rand_number)

        for i in range(len(patients)):
            if i not in drawn_numbers:
                test_set.append(patients[i])

        return learning_set, test_set
