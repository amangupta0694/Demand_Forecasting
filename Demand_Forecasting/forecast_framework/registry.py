# Demand_Forecasting/forecast_framework/registry.py
# Handles discovery and instantiation of forecasting models.

import importlib
import yaml
from omegaconf import OmegaConf, DictConfig
from typing import Union, Any, Dict # Added Dict
import os, sys
import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
_logger.addHandler(handler)

_logger.propagate = False

class ModelRegistry:

    def __init__(self, config: DictConfig):
        """
        Initializes the ModelRegistry.

        Args:
            config: The merged OmegaConf DictConfig object containing both
                    forecasting and model configurations. Expected structure:
                    {
                        'forecasting': {...},
                        'models': { 'ModelName1': {...}, 'ModelName2': {...} }
                    }
        """
        if not isinstance(config, DictConfig):
             raise TypeError(f"Registry expects a DictConfig, got {type(config)}")

        # Store the entire merged config. Models might need access to forecasting params.
        self.conf = config
        _logger.info("ModelRegistry initialized.")
        # _logger.debug(f"Registry received config:\n{OmegaConf.to_yaml(config)}") # Uncomment for debugging

    def get_model(self, model_name: str) -> Any:
        """
        Imports and instantiates the specified forecasting model class.

        Args:
            model_name: The name of the model class to instantiate (e.g., "SKTimeProphet").
                        This name should correspond to a key in the `models` section
                        of the configuration and the actual class name in the model module.

        Returns:
            An instantiated forecasting model object.

        Raises:
            ImportError: If the model module or class cannot be found.
            KeyError: If the model configuration is missing in the `models` section.
            Exception: For other instantiation errors.
        """
        # Construct the expected module path (e.g., forecast_framework.models.sktimeprophet_model)
        # Assuming model file names are lowercase versions of the class name + '_model.py'
        model_module_path = f"forecast_framework.models.{model_name.lower()}_model"
        model_class_name = model_name # The actual class name to import

        try:
            # Check if model-specific config exists
            if model_name not in self.conf.models:
                raise KeyError(f"Configuration for model '{model_name}' not found in the 'models' section of the config.")

            _logger.info(f"Attempting to import model: {model_class_name} from {model_module_path}")
            module = importlib.import_module(model_module_path)
            cls = getattr(module, model_class_name)

            # Pass the *entire* merged configuration to the model's constructor.
            # The model's __init__ is responsible for extracting the parameters it needs
            # from both the 'forecasting' and its specific 'models[model_name]' sections.
            model_instance = cls(self.conf)
            _logger.info(f"Successfully instantiated model: {model_name}")
            return model_instance
        except ModuleNotFoundError:
            _logger.error(f"Model module not found: {model_module_path}")
            raise ImportError(f"Could not find module for model '{model_name}'. Expected at '{model_module_path}.py'")
        except AttributeError:
             _logger.error(f"Model class '{model_class_name}' not found in module '{model_module_path}'")
             raise ImportError(f"Could not find class '{model_class_name}' in module '{model_module_path}'")
        except KeyError as e:
             _logger.error(f"Configuration error for model '{model_name}': {e}")
             raise # Re-raise KeyError
        except Exception as e:
            _logger.error(f"Failed to instantiate model '{model_name}': {e}")
            raise # Re-raise other exceptions

