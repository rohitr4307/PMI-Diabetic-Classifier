import joblib
import numpy as np

def predict(data):
    clf = joblib.load("best_model.joblib")
    return clf.predict(data)

print(predict(np.array([[2, 200, 250, 20, 100, 30, 0.5, 30]])))
