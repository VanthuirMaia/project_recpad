from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

RANDOM_STATE = 42

def get_svm_model():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            method="sigmoid", cv=3
        ))
    ])
    return pipeline


def get_rf_model():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("rf", CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=400, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1
            ),
            method="isotonic", cv=3
        ))
    ])
    return pipeline


def get_mlp_model():
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", CalibratedClassifierCV(
            estimator=MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu", alpha=1e-3,
                max_iter=500, random_state=RANDOM_STATE
            ),
            method="sigmoid", cv=3
        ))
    ])
    return pipeline
