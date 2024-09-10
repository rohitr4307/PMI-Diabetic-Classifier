import joblib

def predict(data):
    clf = joblib.load("/workspaces/PMI-Diabetic-Classifier/best_model.joblib")
    return clf.predict(data)
