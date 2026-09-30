# Iris Classifier with Flask

A small end-to-end machine-learning exercise: train a Random Forest on four flower measurements, save the full preprocessing pipeline and serve predictions through a Flask form.

## Run locally

```bash
python -m venv .venv
# Activate .venv for your shell.
python -m pip install -r requirements.txt
python train.py
python flask-app.py
```

Open `http://127.0.0.1:5000`. Enter sepal length, sepal width, petal length and petal width in centimetres. Decimal values are accepted.

## Implementation

`train.py` uses a fixed, stratified 70/30 split of the bundled `iris.csv`. A scikit-learn `Pipeline` saves the fitted scaler together with the classifier, so prediction uses the same transformation as training. `main.py` remains an alias for training. The web form reads features by name rather than submission order and rejects missing, nonfinite and nonpositive inputs.

Train your own `model.pkl`; the generated pickle is excluded from source control. Only load pickle files you trust. The app binds to localhost and does not enable Flask debug mode by default.

## Verify

```bash
python -m unittest discover -s tests -v
```

Tests exercise decimal input, reordered form fields, invalid requests and agreement with the saved pipeline. The training command prints held-out accuracy for its fixed split. This learning exercise is not an independent benchmark or a production model service.

The original HTML template credits [this CodePen](https://codepen.io/frytyler/pen/EGdtg); that attribution is retained in the template.
