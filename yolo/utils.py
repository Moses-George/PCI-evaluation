import os
import argparse
import sys
import pandas as pd

parent_folder = os.path.abspath((os.path.join(os.path.dirname(__file__), "../")))
sys.path.append(parent_folder)

from dv_prediction.polynomial_fitting.fit_curves import (
    build_all_models,
    predict_deduct_value,
)

normalized_classes = {"alligator": [""], "linear": [""], "pothole": [""]}

classes = list(normalized_classes.keys())


def normalizeClass():
    # keys, vals =
    return


predictions = [
    {"distress_type": "alligator", "severity": "low"},
    {"distress_type": "pothole", "severity": "low"},
    {"distress_type": "linear", "severity": "low"},
    {"distress_type": "alligator", "severity": "medium"},
    {"distress_type": "pothole", "severity": "medium"},
    {"distress_type": "alligator", "severity": "high"},
    {"distress_type": "linear", "severity": "medium"},
    {"distress_type": "pothole", "severity": "high"},
    {"distress_type": "linear", "severity": "high"},
    {"distress_type": "alligator", "severity": "low"},
    {"distress_type": "alligator", "severity": "medium"},
    {"distress_type": "pothole", "severity": "low"},
    {"distress_type": "pothole", "severity": "medium"},
    {"distress_type": "linear", "severity": "low"},
    {"distress_type": "alligator", "severity": "high"},
    {"distress_type": "linear", "severity": "medium"},
    {"distress_type": "pothole", "severity": "high"},
    {"distress_type": "linear", "severity": "high"},
    {"distress_type": "alligator", "severity": "low"},
]


def calculateDeductValue(section_area):
    # Count occurrences of each (distress_type, severity) combination
    count_map = {}
    for prediction in predictions:
        distress_type = prediction.get("distress_type")
        severity = prediction.get("severity")
        if distress_type not in classes:
            print(f"Distress type: {distress_type} not valid")
            continue
        key = (distress_type, severity)
        count_map[key] = count_map.get(key, 0) + 1

    # Build the grouped result with the required keys
    grouped_predictions = []
    for (distress_type, severity), count in count_map.items():
        density = (count * 100) / section_area
        density = round(density, 4)
        models = build_all_models(degree=3, verbose=True)
        dv = predict_deduct_value(models, distress_type, severity, density)
        grouped_predictions.append(
            {
                "distress_type": distress_type,
                "severity": severity,
                "count": count,
                "density": density,
                "deduct_value": round(dv.item(), 4),
            }
        )
        # df = pd.DataFrame()

    return grouped_predictions


def createReportDataFrame():
    return


def clculteTotlDV(): 
    return


if __name__ == "__main__":
    distress_result_table = calculateDeductValue(300)
    dv_values = list(map(lambda result: result["deduct_value"], distress_result_table))
    print(sum(dv_values))
    print(dv_values)
    # for item in distress_result_table:
    #     print(item)
