from pyspark.sql import SparkSession, DataFrame
import pandas as pd

def _multiple_checks(group_df: pd.DataFrame, conf: dict, max_date: pd.Timestamp) -> pd.DataFrame:
    # Example: drop series with too many missing
    threshold = conf.get("missing_threshold", 0.2)
    total = len(group_df)
    missing = group_df[conf["target"]].isna().sum()
    if (missing / total) > threshold:
        return pd.DataFrame(columns=group_df.columns)
    return group_df

class DataQualityChecks:
    def __init__(self, df: DataFrame, conf: dict, spark: SparkSession):
        self.spark = spark
        self.conf = conf
        self.df_pd = df.toPandas()

    def run(self) -> (DataFrame, list):
        # mandatory checks
        if self.conf.get("backtest_length") < self.conf.get("prediction_length"):
            raise ValueError("backtest_length must be >= prediction_length")
        # optional checks
        if self.conf.get("data_quality_check", False):
            max_date = pd.to_datetime(self.df_pd[self.conf["date_col"]]).max()
            clean = self.df_pd.groupby(self.conf["group_id"]).apply(
                lambda g: _multiple_checks(g, self.conf, max_date)
            ).reset_index(drop=True)
        else:
            clean = self.df_pd
        removed = list(set(self.df_pd[self.conf["group_id"]]) - set(clean[self.conf["group_id"]]))
        if clean.empty:
            raise ValueError("No series passed data quality checks")
        spark_df = self.spark.createDataFrame(clean)
        return spark_df, removed
