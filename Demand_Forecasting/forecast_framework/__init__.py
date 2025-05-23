# Demand_Forecasting/forecast_framework/__init__.py
# Initializes the forecast_framework package and defines the main entry point.

from pyspark.sql import SparkSession
from omegaconf import DictConfig # Import DictConfig
import os, sys
import logging # Added logging

# --- Path Setup (Redundant if run_prophet.py handles it, but safe fallback) ---
# Ensure the project root is discoverable if this package is imported differently.
try:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(this_dir) # Assumes forecast_framework is directly under project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # print(f'Project root from forecast_framework/__init__.py: {project_root}') # Debug print
except NameError:
    # Handle cases where __file__ is not defined
    pass

# --- Main Entry Point ---
# Import Forecaster after potential path setup
from forecast_framework._Forecast import Forecaster
import sys
import logging
from pyspark.sql import SparkSession

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
_logger.addHandler(handler)

_logger.propagate = False

def run_forecast(
    spark: SparkSession,
    config: DictConfig, # Accept the merged config object
    run_id: str = None,
) -> str:
    """
    Executes the forecasting pipeline based on the provided configuration.

    Args:
        spark: The active SparkSession.
        config: An OmegaConf DictConfig object containing merged forecasting
                and model configurations. Expected structure:
                {
                    'forecasting': {...}, # Params from forecasting_conf.yaml
                    'models': {...}       # Params from models_conf.yaml
                }
        run_id: An optional existing run ID. If None, a new one is generated.

    Returns:
        The MLflow run ID used for the execution.
    """
    _logger.info("Initializing Forecaster...")
    # The Forecaster class now expects the structured config
    forecaster = Forecaster(spark=spark, config=config, run_id=run_id)
    _logger.info("Starting Forecaster run...")
    return forecaster.run()

# Expose key components if needed, though run_forecast is the primary entry point
__all__ = ["run_forecast", "Forecaster"]

