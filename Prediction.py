import joblib

def predict(data):
    clf = joblib.load("/pmi-diabetic-classifier/master/output/best_model_v1.joblib")
    return clf.predict(data)
