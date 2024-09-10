import joblib

def predict(data):
    clf = joblib.load("/home/runner/work/PMI-Diabetic-Classifier/PMI-Diabetic-Classifier/best_model.joblib")
    return clf.predict(data)
