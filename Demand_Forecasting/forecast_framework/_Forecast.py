# Demand_Forecasting/forecast_framework/_Forecast.py
# Defines the main Forecaster class orchestrating the process.

import os, sys
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import re

import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, lit, concat_ws
import pyspark.sql.functions as F
import pyspark.sql.types as T
from omegaconf import OmegaConf, DictConfig

# Import framework components
from forecast_framework.data_quality import DataQualityChecks
from forecast_framework.registry import ModelRegistry
from forecast_framework.abstract_model import ForecastingRegressor

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
_logger.addHandler(handler)

_logger.propagate = False


class Forecaster:
    def __init__(self, spark: SparkSession, config: DictConfig, run_id: str = None):
        self.spark = spark
        self.conf = config  # Store the merged config
        self.run_id = run_id or str(uuid.uuid4())
        self.model_registry = ModelRegistry(self.conf)

    def _resolve_source(self, config_key: str) -> DataFrame:
        try:
            table_name = self.conf.forecasting[config_key]
            # Quote catalog, database, and table
            pattern = re.compile(r"\b([^.]+)\.([^.]+)\.([^.]+)\b")
            quoted = pattern.sub(r"`\1`.`\2`.`\3`", table_name)
            _logger.info(f"Reading Delta table: {quoted}")
            return self.spark.read.table(quoted)
        except KeyError:
            _logger.error(f"Config key '{config_key}' not found under 'forecasting'")
            raise

    def _generate_forecasts(self, df_input: DataFrame, group_id_col: str, date_col: str, **kwargs) -> DataFrame:
        # Extract config values

        self.spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        self.spark.conf.set("spark.sql.adaptive.enabled", "false")

        group_id_col = self.group_id_col
        date_col = self.date_col
        conf_dict = OmegaConf.to_container(self.conf, resolve=True)

        # Define the UDF schema
        result_schema = T.StructType([
            T.StructField("group_id_col", T.StringType(), True),
            T.StructField("model_name", T.StringType(), True),
            T.StructField("ds", T.TimestampType(), True),
            T.StructField("y_pred", T.DoubleType(), True),
        ])

        def forecast_udf(pdf: pd.DataFrame) -> pd.DataFrame:
            # Create fresh config & registry inside UDF
            udf_config = OmegaConf.create(conf_dict)
            udf_registry = ModelRegistry(udf_config)
            curr_gid = self.group_id_col
            curr_date = self.date_col

            # Handle empty groups
            if pdf.empty:
                return pd.DataFrame(columns=[f.name for f in result_schema])

            group_val = pdf[curr_gid].iloc[0]
            all_forecasts = []

            for model_name in udf_config.forecasting.active_models:
                try:
                    model: ForecastingRegressor = udf_registry.get_model(model_name)
                    pdf_sorted = pdf.sort_values(by=curr_date)
                    model.fit(pdf_sorted.copy())
                    forecast_df, _ = model.forecast(pdf_sorted.copy())

                    # Validate required columns
                    if 'ds' not in forecast_df.columns or 'y_pred' not in forecast_df.columns:
                        _logger.error(
                            f"[Group={group_val}, Model={model_name}] Missing 'ds' or 'y_pred'"
                        )
                        continue

                    # Attach metadata
                    forecast_df['model_name'] = model_name
                    forecast_df[curr_gid] = group_val

                    # Select and cast
                    final_cols = [curr_gid, 'model_name', 'ds', 'y_pred']
                    forecast_df = forecast_df[final_cols]
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                    forecast_df['y_pred'] = pd.to_numeric(forecast_df['y_pred'], errors='coerce')

                    all_forecasts.append(forecast_df)
                except Exception as e:
                    _logger.error(
                        f"Error forecasting Group={group_val}, Model={model_name}: {e}"
                    )

            # Combine or return empty
            if not all_forecasts:
                return pd.DataFrame(columns=[f.name for f in result_schema])

            return pd.concat(all_forecasts, ignore_index=True)

        # Apply the UDF
        _logger.info(f"Applying forecasting UDF grouped by '{group_id_col}'...")
        result_df = df_input.groupBy(group_id_col).applyInPandas(forecast_udf, schema=result_schema)
        _logger.info("Forecasting UDF complete.")
        return result_df

    def run(self) -> str:
        """Orchestrates the forecasting process."""
        _logger.info(f"Starting forecast run ID: {self.run_id}")

        # Extract config
        store_col = self.conf.forecasting.store_id
        item_col = self.conf.forecasting.item_id
        date_col = self.conf.forecasting.date_col
        group_id_col = self.conf.forecasting.group_id
        input_key = 'train_data'
        output_table = self.conf.forecasting.scoring_output

        _logger.info(f"Checking values for store_col: {store_col}, item_col : {item_col}, group_id_col = {group_id_col}, input_key = {input_key}, output_table : {output_table}")

        # 1. Load Data
        try:
            df_raw = self._resolve_source(input_key)

            _logger.info(f"Loaded raw data: {self.conf.forecasting[input_key]} and the count is : {df_raw.count()}")
        except Exception as e:
            _logger.error(f"Failed to load '{input_key}': {e}")
            raise

        # 2. Create Group ID
        if store_col not in df_raw.columns or item_col not in df_raw.columns:
            raise ValueError(
                f"Missing columns for group_id: {store_col}, {item_col}"
            )

        df_with_gid = df_raw.withColumn(
           group_id_col,
            concat_ws("_", col(store_col), col(item_col))
        )

        _logger.info(f"Final columns in the loaded raw tables are: {df_with_gid.columns}")

        # 3. Data Quality Checks
        df_processed = df_with_gid
        if self.conf.forecasting.get('data_quality_check', False):
            _logger.info("Running data quality checks...")
            dq = DataQualityChecks(df=df_processed, conf=self.conf.forecasting, spark=self.spark)
            df_processed, removed = dq.run()
            if removed:
                _logger.warning(f"Removed groups: {removed}")

        # 4. Generate Forecasts
        _logger.info("Generating forecasts...")
        df_forecasts = self._generate_forecasts(df_processed, group_id_col, date_col)

        # # 5. Write Output
        # _logger.info(f"Writing forecasts to {output_table}")
        # try:
        #     out = df_forecasts.withColumn('run_id', lit(self.run_id)) \
        #                       .withColumn('forecast_timestamp', F.current_timestamp())
        #     # Quote table name
        #     if '.' in output_table and '`' not in output_table:
        #         pattern = re.compile(r"\b([^.]+)\.([^.]+)\.([^.]+)\b")
        #         output_table = pattern.sub(r"`\1`.`\2`.`\3`", output_table)

        #     out.write.format('delta').mode('overwrite').saveAsTable(output_table)
        #     _logger.info("Write complete.")
        # except Exception as e:
        #     _logger.error(f"Failed to write: {e}")

        _logger.info(f"Forecast run {self.run_id} completed.")
        return self.run_id
