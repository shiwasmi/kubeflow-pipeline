# ============================================================
# Iris Classification Pipeline - Kubeflow Pipelines (KFP)
# ============================================================
# This pipeline demonstrates a simple end-to-end ML workflow
# using the Iris dataset. It consists of three sequential steps:
#   1. Load & explore data
#   2. Train a Logistic Regression model
#   3. Run prediction on a sample input
#
# Prerequisites:
#   - Docker image: ml-pipeline-image:1.0
#   - Built inside Minikube: minikube docker-env --shell powershell | Invoke-Expression
#   - KFP SDK: pip install kfp==2.16.0
# ============================================================

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from kfp import dsl, compiler
from kfp.dsl import component

# Docker image with all required dependencies pre-installed
IMAGE = "ml-pipeline-image:1.0"


# ------------------------------------------------------------
# Step 1: Load Data
# ------------------------------------------------------------
@component(base_image=IMAGE)
def load_data() -> str:
    from sklearn.datasets import load_iris
    import pandas as pd

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    print("Dataset loaded successfully")
    print(f"Shape: {df.shape}")
    print(df.head())

    return "data_loaded"


# ------------------------------------------------------------
# Step 2: Train Model
# ------------------------------------------------------------
@component(base_image=IMAGE)
def train_model(status: str) -> str:
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    data = load_iris()
    X = data.data
    y = data.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    print("Model trained successfully")
    print(f"Classes: {list(data.target_names)}")

    return "model_trained"


# ------------------------------------------------------------
# Step 3: Predict
# ------------------------------------------------------------
@component(base_image=IMAGE)
def predict(status: str):
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    data = load_iris()
    X = data.data
    y = data.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    sample = X[0].reshape(1, -1)
    prediction = model.predict(sample)
    predicted_class = data.target_names[prediction[0]]

    print(f"Sample input: {X[0]}")
    print(f"Predicted class index: {prediction[0]}")
    print(f"Predicted class name: {predicted_class}")


# ------------------------------------------------------------
# Pipeline Definition
# ------------------------------------------------------------
@dsl.pipeline(
    name="iris-classification-pipeline",
    description="End-to-end Iris classification: load data -> train model -> predict"
)
def iris_classification_pipeline():
    step1 = load_data()
    step1.set_caching_options(False)

    step2 = train_model(status=step1.output)
    step2.set_caching_options(False)

    step3 = predict(status=step2.output)
    step3.set_caching_options(False)


# ------------------------------------------------------------
# Compile Pipeline
# ------------------------------------------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=iris_classification_pipeline,
        package_path="iris_classification_pipeline.yaml"
    )
