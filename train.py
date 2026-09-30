"""Train the complete preprocessing + classifier pipeline on the bundled Iris CSV."""
from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
ROOT = Path(__file__).resolve().parent


def train(data_path=ROOT / "iris.csv", model_path=ROOT / "model.pkl"):
    frame = pd.read_csv(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["Class"], test_size=0.3, random_state=50,
        stratify=frame["Class"],
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=50)),
    ])
    model.fit(x_train, y_train)
    with Path(model_path).open("wb") as output:
        pickle.dump(model, output)
    return model, model.score(x_test, y_test), len(x_test)


if __name__ == "__main__":
    _, accuracy, count = train()
    print(f"Held-out accuracy: {accuracy:.3f} on {count} rows (fixed split; not cross-validation)")
