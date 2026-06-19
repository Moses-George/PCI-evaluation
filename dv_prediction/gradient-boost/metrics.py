from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def evaluate_all_models(data_dict, dv_models):
    """
    data_dict format:
    {
        "alligator": df,
        "pothole": df,
        ...
    }
    """

    results = {}

    for distress, df in data_dict.items():

        X = df[["severity", "density"]]
        y = df["DV"]

        model = dv_models[distress]

        metrics = evaluate_model(model, X, y)

        results[distress] = metrics

        print(f"\n{distress.upper()}")
        print(metrics)

    return results