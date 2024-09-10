import joblib

def predict(data):
    clf = joblib.load("best_model.joblib")
    return clf.predict(data)
