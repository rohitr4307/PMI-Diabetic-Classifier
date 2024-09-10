import joblib
import numpy as np
# from .PMI_Diabetic_Classifier import outlier_treatment



def predict(data):
    clf = joblib.load("best_model.joblib")
    return clf.predict(data)