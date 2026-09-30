"""Local Iris classification demo. Load only a model you trained yourself."""
from pathlib import Path
import math
import pickle
import pandas as pd
from flask import Flask, request, render_template
from train import FEATURES, ROOT


def create_app(model_path=ROOT / "model.pkl"):
    app = Flask(__name__)
    path = Path(model_path)
    if not path.exists():
        raise RuntimeError("Run python train.py to generate the local model first.")
    with path.open("rb") as source:
        model = pickle.load(source)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        try:
            values = [float(request.form[field]) for field in FEATURES]
            if not all(math.isfinite(x) and x > 0 for x in values):
                raise ValueError("Measurements must be finite and positive.")
        except (KeyError, TypeError, ValueError):
            return render_template("index.html", prediction_text="Enter four positive numeric measurements in centimetres."), 400
        frame = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(frame)[0]
        return render_template("index.html", prediction_text=f"The flower species is {prediction}")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000)
