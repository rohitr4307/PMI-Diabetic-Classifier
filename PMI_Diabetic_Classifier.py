import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="dark")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from itertools import islice
from joblib import dump, load
from Prediction import predict
# 
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, cross_validate, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, precision_recall_fscore_support, precision_recall_curve, f1_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
# import kaggle

# kaggle.api.authenticate()
# kaggle.api.dataset_download_files('rohitr4307/pima-indians-diabetes-database', path='./data', unzip=True)


# df = pd.read_csv("D:\Training\BIA\Excel\diabetes.csv")

df = df = pd.read_csv("/home/runner/work/PMI-Diabetic-Classifier/PMI-Diabetic-Classifier/diabetes.csv")

def outlier_treatment(df):

    df = df.copy()

    for col in df.columns:

      q3 = df[col].quantile(0.75)
      q1 = df[col].quantile(0.25)
      iqr = q3 - q1

      upper_iqr = q3 + iqr * 1.5
      lower_iqr = q1 - iqr * 1.5

      # print(df.loc[(df[col] > upper_iqr) | (df[col] < lower_iqr)].shape)
      df.loc[(df[col] > upper_iqr) | (df[col] < lower_iqr), col] = df[df[col] > 0.0][col].median()
      # print(df.loc[(df[col] > upper_iqr) | (df[col] < lower_iqr)].shape)

    return df

feature_engineering = FunctionTransformer(outlier_treatment, validate=False)

def modeling_pipeline(df, classifier_name, model, scalers, param_grid, train_x, train_y, test_x, test_y, feature_engineering, n_spilits=5, scoring='f1'):

    # KFold Stratigies
    cv_strats = {
    'KFold': KFold(random_state=11, shuffle=True, n_splits=n_spilits),
    'Stratified KFold': StratifiedKFold(random_state=11, shuffle=True, n_splits=n_spilits)
    }

    # creating pipeline and running GridSearchCV to tune the hyperparameter

    results = {}

    for name, fold_model in cv_strats.items():

        # Scalers to scal the features
        scalers = scalers

        for sclr in scalers:

          smote = SMOTE(random_state=11)

          pipeline = Pipeline([
              ('feature_eng', feature_engineering),
              ('scaler', sclr),
              ('smote', smote),
              # ('feature_selection', SelectKBest(score_func=mutual_info_classif, k=3)),
              ('classifier', model)
          ])

          param_grid = {
              'classifier': [model]
          } if len(param_grid) == 0 else param_grid

          grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=fold_model, scoring=scoring)

          grid_search.fit(train_x, train_y)

          pred = grid_search.best_estimator_.predict(test_x)

          score_functions = {
            'f1': f1_score,
            'roc_auc': roc_auc_score,
            'recall': recall_score,
            'precision': precision_score,
            'accuracy': accuracy_score
          }

          test_score = np.round(score_functions[scoring](test_y, pred), 2)

          print(f"Fold Strategies: {name}, Scaler: {sclr}, Classifier: {classifier_name}, Train {scoring} Score: {np.round(grid_search.best_score_, 2)}, Test {scoring} Score: {test_score}")

          results[name+"_"+str(sclr)+"_"+classifier_name] = {
              'best_params': grid_search.best_params_,
              'best_model': grid_search.best_estimator_,
              'best_score_train': np.round(grid_search.best_score_, 2),
              'best_score_test': test_score,
              'test_pred': pred
          }

    return results

# Splitting data into train and test
train_x, test_x, train_y, test_y = train_test_split(df.drop('Outcome', axis=1), df['Outcome'], train_size=0.8,
                                                    random_state=11, shuffle=True, stratify=df['Outcome'])

# scalers = [MinMaxScaler(), StandardScaler(), RobustScaler()]

# classifiers = {
#     "Logistic Regression": LogisticRegression(),
#     "SVC": SVC(),
#     "KNeighborsClassifier": KNeighborsClassifier(),
#     "DecisionTreeClassifier": DecisionTreeClassifier(),
#     "RandomForestClassifier": RandomForestClassifier(),
#     "GradientBoostingClassifier": GradientBoostingClassifier(),
#     "AdaBoostClassifier": AdaBoostClassifier(),
#     "XGBClassifier": XGBClassifier()
# }

# param_grid = {}

scalers = [RobustScaler()]

classifiers = {
    "RandomForestClassifier": RandomForestClassifier()
}

param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20, 30],   # Maximum depth of the tree
    'classifier__min_samples_split': [2, 5, 10],           # Minimum number of samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4, 6, 8],       # Minimum number of samples required at each leaf node
    # 'classifier__max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider when looking for the best split
    # 'classifier__bootstrap': [True, False],                # Whether bootstrap samples are used when building trees
    # 'classifier__criterion': ['gini', 'entropy'],          # Function to measure the quality of a split
    'classifier__class_weight': [None, 'balanced', 'balanced_subsample'], # Weights associated with classes
    'classifier__random_state': [11],                      # Ensures reproducibility (use the same seed across different experiments)
    # 'classifier__ccp_alpha': [0.0, 0.01, 0.1],             # Complexity parameter used for Minimal Cost-Complexity Pruning
    'classifier__max_samples': [None, 0.5, 0.75, 1.0]      # Number of samples to draw for training each base estimator (if bootstrap=True)
}

final_result = {}
for name, model in classifiers.items():
  results = modeling_pipeline(df, name, model, scalers, {}, train_x, train_y, test_x, test_y, feature_engineering, n_spilits=8, scoring='f1')
  final_result.update(results)

best_result = list(islice(final_result.items(), 1))[0]
best_model = best_result[1]['best_model']
pred_score = best_result[1]['test_pred']

dump(best_model, 'best_model.joblib')

# test = pd.DataFrame(data=[[2, 200, 250, 20, 100, 30, 0.5, 30]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#       'BMI', 'DiabetesPedigreeFunction', 'Age'])
# print(test.columns)
# print("Output", predict(test)[0])
