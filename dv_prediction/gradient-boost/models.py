from data_prep import severity_levels, load_astm_curve, severity_map, DISTRESSES
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from metrics import evaluate_all_models

dv_models = {}
data_dict = {}

def train_models():

    for distress in DISTRESSES:

        all_data = []

        for sev in severity_levels:
            df = load_astm_curve(distress, sev)

            df["severity"] = severity_levels.index(sev)
            data_dict[distress] = df
            all_data.append(df)

        full_df = pd.concat(all_data, ignore_index=True)

        X = full_df[["severity", "density"]]
        y = full_df["DV"]

        model = GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3
        )

        model.fit(X, y)

        dv_models[distress] = model

        print(f"[OK] ASTM-trained model: {distress}")


if __name__ == "__min__":
    train_models()

    results = evaluate_all_models(data_dict)

    # def predict_dv(distress, severity, density):

    #     model = dv_models[distress]

    #     x = np.array([[severity_map[severity], density]])

    #     dv = model.predict(x)[0]

    #     return float(np.clip(dv, 0, 100))

    # samples = [
    #     {"distress": "alligator", "severity": "low", "density": 12},
    #     {"distress": "pothole", "severity": "high", "density": 8},
    #     {"distress": "transverse", "severity": "medium", "density": 40},
    # ]

    # for s in samples:
    #     print(s, "→ DV:", predict_dv(s["distress"], s["severity"], s["density"]))
