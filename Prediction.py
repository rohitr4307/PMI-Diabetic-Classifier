import joblib
import numpy as np

def predict(data):
    clf = joblib.load("best_model.joblib")
    return clf.predict(data)