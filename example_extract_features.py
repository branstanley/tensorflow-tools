from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType, DateType, IntegerType
import pandas as pd
import numpy as np
from FeatureEngineering.compute_features_v2 import compute_features_v2
from FeatureEngineering.compute_categorical_features_v1 import compute_categorical_features_v1

# Define output schema
feature_schema = StructType([
    StructField("Key1", StringType()),
    StructField("Key2", StringType()),
    StructField("features", ArrayType(DoubleType()))
])

def extract_features(pdf: pd.DataFrame) -> pd.DataFrame:
    
    start_of_cosine_cycle = pdf["cosine_cycle"].iloc[0]
    end_of_cosine_cycle = pdf["cosine_cycle"].iloc[-1]
    start_of_sine_cycle = pdf["sine_cycle"].iloc[0]
    end_of_sine_cycle = pdf["sine_cycle"].iloc[-1]

    # One-hot encoding (replace with actual categorical columns)
    status_categories = ["on", "off"]

    for status in status_categories:
        pdf[f"status_{status}"] = ((pdf["status"].notnull()) & (pdf["status"] == status)).astype(float)
        pdf[f"status_{status}"] = pdf[f"status_{status}"].fillna(0.0)

    # Group by device
    grouped = []

    for (dwelling, Key2), group in pdf.groupby(["Key1", "Key2"]):
        feature_vector = [start_of_cosine_cycle, end_of_cosine_cycle, start_of_sine_cycle, end_of_sine_cycle]
        # Calculate features for each numeric column (replace with actual numeric columns)
        feature_vector += compute_features_v2(group["someValue"])
        
        # Calculate features for each one hot encoded column (replace with actual categorical columns)
        for status in status_categories:
            feature_vector += compute_categorical_features_v1(group[f"status_{status}"])

        feature_vector = [float(f) if isinstance(f, (int, float, np.number)) and np.isfinite(f) else 0.0 for f in feature_vector]
        grouped.append((dwelling, Key2, feature_vector))

    return pd.DataFrame(grouped, columns=["Key1", "Key2", "features"])