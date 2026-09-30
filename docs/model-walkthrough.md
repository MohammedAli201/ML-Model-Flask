# Trace one Iris prediction

The project connects a small scikit-learn model to a Flask form. Its main engineering concern is keeping the feature contract consistent from training to inference.

## Training

[train.py](../train.py) selects four named measurements from the bundled CSV. It makes a stratified 70/30 split with random state 50, fits a Pipeline containing StandardScaler and a RandomForestClassifier, and saves the fitted pipeline.

The scaler fits inside the pipeline using the training split. The same saved transformation is applied at inference. Scaling is retained as a pipeline example; a Random Forest does not require scaling in the way a distance-based model does.

The command prints accuracy for the fixed held-out split. That result is useful for checking the exercise, but it is not cross-validation or an independent benchmark.

## Inference

[flask-app.py](../flask-app.py) loads a locally trained model at startup. POST `/predict` reads values using the named feature contract:

| Field | Meaning | Unit |
| --- | --- | --- |
| `Sepal_Length` | Sepal length | cm |
| `Sepal_Width` | Sepal width | cm |
| `Petal_Length` | Petal length | cm |
| `Petal_Width` | Petal width | cm |

Reading by name makes submission order irrelevant. Missing, malformed, nonfinite and nonpositive values return HTTP 400. Accepted values become a DataFrame with the same column names used during training.

## Reproduce one request

After `python train.py` and `python flask-app.py`:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -d 'Sepal_Length=5.1' -d 'Sepal_Width=3.5' \
  -d 'Petal_Length=1.4' -d 'Petal_Width=0.2'
```

The response is an HTML page containing the predicted species.

## Verification and boundaries

[tests/test_prediction.py](../tests/test_prediction.py) trains a temporary model and checks the Flask response against a direct prediction from that saved pipeline. It also checks invalid inputs and home-page rendering. The tests do not need an external API.

Generated pickle files remain local. Only load files you trained or otherwise trust. The server binds to localhost with debug disabled; it is a demonstration, not a deployed model service. The original template attribution remains intact.
