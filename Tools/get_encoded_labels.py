from sklearn.preprocessing import LabelEncoder
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd

def get_encoded_labels(df, grouping_cols, feature_col="features"):
    """
    Generate encoded labels for classification based on specified grouping columns.
    Args:
        df: Input Spark DataFrame.
        grouping_cols: Column name or list of column names to group by.
        feature_col: Name of the column containing feature vectors (e.g., 'features').
    Returns:
        num_classes: Number of unique classes.
        classifier_input_values: Feature values for classifier input.
        classifier_input_labels: Encoded label array for classifier input.
    """
    if isinstance(grouping_cols, str):
        grouping_cols = [grouping_cols]

    windowSpec = Window.orderBy(*grouping_cols)

    classifier_df = df\
        .select(*grouping_cols)\
        .distinct()\
        .withColumn("GlobalId", F.row_number().over(windowSpec))

    features_df = df\
        .join(F.broadcast(classifier_df), on=grouping_cols, how="inner")

    classifier_pdf = features_df.toPandas()

    input_val = classifier_pdf[feature_col].apply(pd.Series)
    input_label = classifier_pdf["GlobalId"]

    classifier_input_values = input_val
    classifier_input_labels = input_label

    encoder = LabelEncoder()
    encoder.fit(input_label)  

    classifier_input_labels = encoder.transform(classifier_input_labels)

    num_classes = len(encoder.classes_)
    return num_classes, classifier_input_values, classifier_input_labels