# Databricks notebook source
import sys
import os
import logging
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, Tuple
from Demand_Forecasting.forecast_framework import run_forecast


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

logger.propagate = False
# --- Configuration Loading ---
def load_config(project_root_path: str) -> DictConfig:
    """Loads and merges configuration files."""
    try:
        # Define paths to config files relative to project root
        forecasting_conf_path = os.path.join(project_root_path, "configs", "forecasting_conf.yaml")
        models_conf_path = os.path.join(project_root_path, "configs", "models_conf.yaml")

        logger.info(f"Loading forecasting config from: {forecasting_conf_path}")
        forecasting_conf = OmegaConf.load(forecasting_conf_path)

        logger.info(f"Loading models config from: {models_conf_path}")
        models_conf = OmegaConf.load(models_conf_path)

        # Create a structured merged configuration
        merged_conf = OmegaConf.create({
            "forecasting": forecasting_conf,
            "models": models_conf
        })

        # Resolve interpolations (like ${forecasting.catalog}.${forecasting.db}...)
        OmegaConf.resolve(merged_conf)
        logger.info("Configurations loaded and merged successfully.")
        logger.info(f"The configurations loaded and merged are :{merged_conf}")
        return merged_conf
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}. Ensure paths are correct relative to project root: {project_root_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading or merging configurations: {e}")
        raise

# COMMAND ----------

if os.path.basename(os.getcwd()) == "Prophet":
         project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
elif os.path.basename(os.getcwd()) == "notebooks":
         project_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
else: # Assume run from project root
         project_root = os.path.abspath(os.getcwd())
        #  project_root = os.path.join("file:/", project_root)
print(f"Determined project root: {project_root}")

# COMMAND ----------


_config = load_config(project_root_path=project_root)
# print(_config)
run_forecast(spark=spark, config = _config, run_id = None)