import joblib

def predict(data):
    clf = joblib.load("/output/best_model.joblib")
    return clf.predict(data)
