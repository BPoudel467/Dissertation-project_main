import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils.common import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR


def plot_confusion_matrix():

    X_test = joblib.load(PROCESSED_DIR / "X_test_selected.pkl")
    y_test = joblib.load(PROCESSED_DIR / "y_test.pkl")

    model = joblib.load(MODELS_DIR / "randomforest.pkl")

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot()

    plt.title("Random Forest Confusion Matrix")

    plt.savefig(FIGURES_DIR / "confusion_matrix_rf.png")

    plt.close()