import joblib
import numpy as np
# from .PMI_Diabetic_Classifier import outlier_treatment



def predict(data):
    clf = joblib.load("output/best_model_v1.joblib")
    return clf.predict(data)
