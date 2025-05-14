from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Parameters:
    - X_train, X_test: Train/test features
    - y_train, y_test: Train/test targets
    - preprocessor: ColumnTransformer for feature processing
    Returns:
    - trained_models (dict): Fitted models
    - results (dict): Accuracy and classification report
    """

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    }

    trained_models = {}
    results = {}

    for name, clf in classifiers.items():
        print(f"[INFO] Training {name}...")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        trained_models[name] = pipeline
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

    return trained_models, results
