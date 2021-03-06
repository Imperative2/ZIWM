from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.svm import SVR


class FeatureSelector:

    @staticmethod
    def selectKBestFeatures( num_of_features, features_array, class_assignment):

        features_array = np.array(features_array)
        features_array = features_array.astype(float)
        class_assignment = np.array(class_assignment)
        class_assignment = class_assignment.astype(float)

        estimator = SVR(kernel="linear")
        selectorRFE = RFE(estimator, num_of_features, step=1)
        selected_features = selectorRFE.fit_transform(features_array, class_assignment)


        # selectorCHI = SelectKBest(chi2, k=num_of_features)
        # selected_features = selectorCHI.fit_transform(features_array, class_assignment)



        print("first sel", selectorRFE)
        print("RFE selector", len(selected_features[0]))
        print("selected features", selectorRFE.ranking_)




        return selected_features