# ================================================
# Housing Price Prediction - Kubeflow Pipeline
# ================================================
# Dataset  : Housing Prices (CSV via direct URL)
# Models   : Linear Regression, Random Forest,
#            Decision Tree
# Steps    : Download → Validate → Feature Eng
#            → Split → Train → Evaluate
# Base image: ml-pipeline-image:1.0
# ================================================

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from kfp import dsl, compiler
from kfp.dsl import component

IMAGE = "ml-pipeline-image:1.0"

# ------------------------------------------------
# COMPONENT 1 : Download Dataset
# ------------------------------------------------
@component(base_image=IMAGE)
def download_dataset(dataset: dsl.OutputPath("Dataset")):  # artifact type
    import requests, pandas as pd, io, numpy as np

    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/housing_prices.csv"
    print("Downloading dataset from:", url)

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print("WARNING: Could not download dataset, using synthetic data:", e)
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "price": np.random.randint(100000, 1000000, n),
            "area": np.random.randint(1000, 10000, n),
            "bedrooms": np.random.randint(1, 6, n),
            "bathrooms": np.random.randint(1, 4, n),
            "stories": np.random.randint(1, 4, n),
            "mainroad": np.random.choice(["yes", "no"], n),
            "guestroom": np.random.choice(["yes", "no"], n),
            "basement": np.random.choice(["yes", "no"], n),
            "hotwaterheating": np.random.choice(["yes", "no"], n),
            "airconditioning": np.random.choice(["yes", "no"], n),
            "parking": np.random.randint(0, 4, n),
            "prefarea": np.random.choice(["yes", "no"], n),
            "furnishingstatus": np.random.choice(
                ["furnished", "semi-furnished", "unfurnished"], n
            ),
        })

    df.to_csv(dataset, index=False)
    print("Dataset ready:", df.shape)


# ------------------------------------------------
# COMPONENT 2 : Validate Data
# ------------------------------------------------
@component(base_image=IMAGE)
def validate_data(dataset: dsl.InputPath("Dataset"),
                  validated: dsl.OutputPath("Dataset")):
    import pandas as pd
    df = pd.read_csv(dataset)
    print("Validating dataset... Shape:", df.shape)

    df = df.fillna(0).drop_duplicates()
    df.to_csv(validated, index=False)
    print("Validation complete. Shape:", df.shape)


# ------------------------------------------------
# COMPONENT 3 : Feature Engineering
# ------------------------------------------------
@component(base_image=IMAGE)
def feature_engineering(validated: dsl.InputPath("Dataset"),
                        features: dsl.OutputPath("Dataset")):
    import pandas as pd
    df = pd.read_csv(validated)
    print("Starting feature engineering...")

    df = pd.get_dummies(df)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    df.to_csv(features, index=False)
    print("Feature engineering complete. Features:", df.shape[1])


# ------------------------------------------------
# COMPONENT 4 : Split Dataset
# ------------------------------------------------
@component(base_image=IMAGE)
def split_data(features: dsl.InputPath("Dataset"),
               train_set: dsl.OutputPath("Dataset"),
               test_set: dsl.OutputPath("Dataset")):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(features)
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv(train_set, index=False)
    test.to_csv(test_set, index=False)

    print("Split complete. Train:", len(train), "Test:", len(test))


# ------------------------------------------------
# COMPONENT 5 : Train Models
# ------------------------------------------------
@component(base_image=IMAGE)
def train_models(train_set: dsl.InputPath("Dataset"),
                 models: dsl.OutputPath("Dataset")):
    import os, pandas as pd, joblib
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    df = pd.read_csv(train_set)
    if "price" not in df.columns:
        raise ValueError("Column 'price' not found. Columns: " + str(df.columns.tolist()))

    X, y = df.drop("price", axis=1), df["price"]

    os.makedirs(models, exist_ok=True)
    joblib.dump(LinearRegression().fit(X, y), os.path.join(models, "linear.joblib"))
    joblib.dump(RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y),
                os.path.join(models, "rf.joblib"))
    joblib.dump(DecisionTreeRegressor(random_state=42).fit(X, y),
                os.path.join(models, "dt.joblib"))

    print("Models trained and saved to", models)


# ------------------------------------------------
# COMPONENT 6 : Evaluate Models
# ------------------------------------------------
@component(base_image=IMAGE)
def evaluate_models(test_set: dsl.InputPath("Dataset"),
                    models: dsl.InputPath("Dataset")):
    import os, pandas as pd, joblib, math
    from sklearn.metrics import mean_squared_error, r2_score

    df = pd.read_csv(test_set)
    X, y = df.drop("price", axis=1), df["price"]

    results = {}
    for file in os.listdir(models):
        if file.endswith(".joblib"):
            name = file.replace(".joblib", "")
            model = joblib.load(os.path.join(models, file))
            pred = model.predict(X)
            rmse = math.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            results[name] = r2
            print(f"{name}: RMSE={rmse:.2f}, R2={r2:.4f}")

    best = max(results, key=results.get)
    print(f"Best model: {best} (R2={results[best]:.4f})")


# ------------------------------------------------
# PIPELINE DEFINITION
# ------------------------------------------------
@dsl.pipeline(
    name="housing-price-prediction-pipeline",
    description="End-to-end housing price prediction pipeline",
)
def housing_pipeline():
    download = download_dataset()
    validate = validate_data(dataset=download.outputs["dataset"])
    features = feature_engineering(validated=validate.outputs["validated"])
    split = split_data(features=features.outputs["features"])
    train = train_models(train_set=split.outputs["train_set"])
    evaluate_models(test_set=split.outputs["test_set"], models=train.outputs["models"])


# ------------------------------------------------
# COMPILE
# ------------------------------------------------
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=housing_pipeline,
        package_path="housing_ml_pipeline.yaml",
    )
    print("housing_ml_pipeline.yaml generated successfully")
