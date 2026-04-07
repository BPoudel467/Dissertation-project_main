import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from src.utils.common import PROCESSED_DIR, MODELS_DIR, TABLES_DIR


def load_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train_selected.pkl")
    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")

    return X_train, X_test, y_test


# -------------------------
# KMEANS CLUSTERING
# -------------------------

def run_kmeans(X_train, X_test, y_test):

    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X_train)

    preds = model.predict(X_test)

    results = {
        "model": "k-Means",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    joblib.dump(model, MODELS_DIR / "kmeans.pkl")

    return results


# -------------------------
# AUTOENCODER
# -------------------------

def run_autoencoder(X_train, X_test, y_test):

    input_dim = X_train.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(input_layer)
    encoded = Dense(16, activation="relu")(encoded)

    decoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse"
    )

    autoencoder.fit(
        X_train,
        X_train,
        epochs=20,
        batch_size=256,
        shuffle=True,
        verbose=0
    )

    reconstructions = autoencoder.predict(X_test)

    errors = np.mean(np.square(X_test - reconstructions), axis=1)

    threshold = np.percentile(errors, 95)

    preds = (errors > threshold).astype(int)

    results = {
        "model": "Autoencoder",
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }

    autoencoder.save(MODELS_DIR / "autoencoder.h5")

    return results


def run_unsupervised():

    X_train, X_test, y_test = load_data()

    results = []

    results.append(run_kmeans(X_train, X_test, y_test))
    results.append(run_autoencoder(X_train, X_test, y_test))

    df = pd.DataFrame(results)

    df.to_csv(TABLES_DIR / "unsupervised_results.csv", index=False)

    print(df)