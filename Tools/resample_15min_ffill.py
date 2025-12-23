from pyspark.sql import functions as F
from pyspark.sql.window import Window

def resample_15min_ffill(df):
    # Step 1: Round timestamps down to the nearest 15-minute interval
    resampled_df = df.withColumn(
        "quarter_ts",
        F.from_unixtime((F.floor(F.unix_timestamp("timestamp") / 900) * 900)).cast("timestamp")
    )

    # Step 2: Aggregate per 15 minutes per group
    resampled_df = resampled_df\
    .groupBy("ADD YOUR GROUP BYS HERE", "quarter_ts")\
    .agg(
        F.avg("Some Numerical Data").alias("Some Numerical Data"),
        F.last("Some Categorical Data").alias("Some Categorical Data")
    )

    # Step 3: Create a complete 15-minute time series per group
    time_bounds = resampled_df.groupBy("ADD YOUR GROUP BYS HERE").agg(
        F.min("quarter_ts").alias("min_ts"),
        F.max("quarter_ts").alias("max_ts")
    )

    time_series = time_bounds.withColumn(
        "quarter_ts",
        F.explode(F.sequence("min_ts", "max_ts", F.expr("INTERVAL 15 MINUTES")))
    )

    resampled_df = time_series.join(
        resampled_df,
        on=["ADD YOUR GROUP BYS HERE", "quarter_ts"],
        how="left"
    )

    # Step 4: Forward fill missing values
    windowSpec = Window.partitionBy("ADD YOUR GROUP BYS HERE").orderBy("quarter_ts").rowsBetween(Window.unboundedPreceding, 0)

    return resampled_df.select(
        F.col("ADD YOUR GROUP BYS HERE"),
        F.col("quarter_ts").alias("timestamp"),
        F.last("Some Numerical Data", ignorenulls=True).over(windowSpec).alias("Some Numerical Data"),
        F.last("Some Categorical Data", ignorenulls=True).over(windowSpec).alias("Some Categorical Data")
    )