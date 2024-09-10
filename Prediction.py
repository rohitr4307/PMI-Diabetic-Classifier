import joblib

def predict(data):
    clf = joblib.load("best_model_v1.joblib")
    return clf.predict(data)
