# Demand_Forecasting/forecast_framework/abstract_model.py
# Defines the abstract base class for all forecasting models.

from abc import ABC, abstractmethod # Use ABC for abstract base class
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from omegaconf import DictConfig # Import DictConfig
import logging
from typing import Tuple

# Import performance metric - ensure sktime is installed
try:
    from sktime.performance_metrics.forecasting import smape_loss
except ImportError:
    logging.warning("sktime not found, smape_loss unavailable. Install sktime for backtesting metrics.")
    # Define a placeholder if sktime is optional
    def smape_loss(y_true, y_pred):
        return np.nan # Or raise an error

_logger = logging.getLogger(__name__)

class ForecastingRegressor(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract Base Class for forecasting models in the framework.
    Requires subclasses to implement fit, forecast, and prepare_data.
    Provides a default backtesting implementation.
    """
    def __init__(self, config: DictConfig):
        """
        Initializes the base forecasting regressor.

        Args:
            config: The merged OmegaConf DictConfig object containing both
                    forecasting and model-specific parameters.
        """
        if not isinstance(config, DictConfig):
             raise TypeError(f"Model expects a DictConfig, got {type(config)}")
        # Store the entire config; subclasses extract what they need.
        self.params = config
        # Extract general forecasting parameters needed by the base class (e.g., for backtesting)
        # These are expected to be under the 'forecasting' key.
        try:
            forecasting_params = self.params.forecasting
            self.freq = forecasting_params.get("freq", "D").upper()[0] # Default to Daily
            self.prediction_length = forecasting_params.prediction_length
            self.stride = forecasting_params.get("stride", 1) # Default stride to 1 if not provided
            self.date_col = forecasting_params.date_col
            self.group_id = forecasting_params.group_id
            self.target = forecasting_params.target
            self.metric = forecasting_params.get("metric", "smape") # Default metric

        except AttributeError as e:
            _logger.error(f"Missing essential key in 'forecasting' section of config: {e}")
            raise ValueError(f"Configuration missing essential forecasting parameters: {e}")
        except Exception as e:
             _logger.error(f"Error accessing forecasting parameters from config: {e}")
             raise

        # Calculate time offsets based on frequency
        self.one_ts_offset = self._calculate_offset(1)
        self.prediction_length_offset = self._calculate_offset(self.prediction_length)
        self.stride_offset = self._calculate_offset(self.stride)

        if self.one_ts_offset is None or self.prediction_length_offset is None or self.stride_offset is None:
             _logger.error(f"Unsupported frequency '{self.freq}' for offset calculation.")
             raise ValueError(f"Unsupported frequency: {self.freq}")


    def _calculate_offset(self, periods: int) -> pd.DateOffset:
        """ Calculates pandas DateOffset based on frequency and number of periods. """
        if self.freq == "M":
            return pd.offsets.MonthEnd(periods)
        elif self.freq == "W":
            # Use DateOffset for weeks to avoid anchoring to specific weekday like WeekOfMonth
            return pd.DateOffset(weeks=periods)
        elif self.freq == "D":
            return pd.DateOffset(days=periods)
        elif self.freq == "H":
            return pd.DateOffset(hours=periods)
        # Add other frequencies (Q, Y, etc.) if needed
        else:
            return None # Indicate unsupported frequency


    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert and clean pandas DataFrame specific to the model's requirements.
        Subclasses must implement this.

        Args:
            df: Input pandas DataFrame for a single group.

        Returns:
            Processed pandas DataFrame suitable for the model's fit/forecast methods.
        """
        pass

    @abstractmethod
    def fit(self, train_df: pd.DataFrame):
        """
        Fit the forecasting model on historical data.
        Subclasses must implement this.

        Args:
            train_df: The training data (pandas DataFrame) for a single group,
                      potentially pre-processed by `prepare_data`.

        Returns:
            self: The fitted model instance.
        """
        pass

    @abstractmethod
    def forecast(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """
        Generate forecasts for the prediction length horizon.
        Subclasses must implement this.

        Args:
            df: The historical data (pandas DataFrame) needed to generate
                the forecast (e.g., for creating future regressors or context).

        Returns:
            A tuple containing:
            - forecast_df (pd.DataFrame): DataFrame with columns like 'unique_id',
              'ds' (timestamp), 'y_pred' (forecast value).
            - model_instance (object): The fitted model object used for forecasting.
        """
        pass

    def backtest(self, df: pd.DataFrame, start: pd.Timestamp) -> list:
        """
        Perform backtesting over sliding windows using the model's fit and forecast methods.

        Args:
            df: The historical data (pandas DataFrame) for a single group.
            start: The timestamp indicating the start of the very first training window's
                   *prediction* period. The training data will be all data *before* this.

        Returns:
            A list of tuples, where each tuple represents the evaluation result for one window:
            (unique_id, model_name, metric_name, metric_value)
        """
        if smape_loss is np.nan: # Check if metric function is available
             _logger.warning("Skipping backtest calculation because sktime.performance_metrics.smape_loss is unavailable.")
             return []

        # Ensure date column is datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(by=self.date_col) # Ensure data is sorted by time

        # Backtesting parameters from instance variables (set in __init__)
        group_id_val = df[self.group_id].iloc[0]
        model_name = type(self).__name__ # Get the name of the concrete class

        # Dates for iteration
        # `start` is the beginning of the first *test* set.
        curr_test_start_date = pd.Timestamp(start)
        # The end date of the full historical dataset
        max_hist_date = df[self.date_col].max()

        results = []
        window_num = 0

        # Loop while the *end* of the next test window is within the historical data range
        while curr_test_start_date + self.prediction_length_offset <= max_hist_date + self.one_ts_offset:
            window_num += 1
            # Define training data: all data strictly *before* the current test window starts
            train_df = df[df[self.date_col] < curr_test_start_date].copy()

            # Define actuals data: data within the current test window
            test_window_end_date = curr_test_start_date + self.prediction_length_offset
            actuals_df = df[(df[self.date_col] >= curr_test_start_date) &
                            (df[self.date_col] < test_window_end_date)].copy() # Use < for end date

            if train_df.empty:
                 _logger.warning(f"Backtest window {window_num} for group {group_id_val}: Training data is empty. Skipping window.")
                 # Move to the next window
                 curr_test_start_date += self.stride_offset
                 continue

            if actuals_df.empty or len(actuals_df) < self.prediction_length:
                _logger.warning(f"Backtest window {window_num} for group {group_id_val}: Insufficient actuals data ({len(actuals_df)} points) for prediction length ({self.prediction_length}). Skipping window.")
                # Move to the next window
                curr_test_start_date += self.stride_offset
                continue

            try:
                # Fit the model on the current training window
                self.fit(train_df) # Fit method should return self

                # Generate forecast using the historical data context (train_df)
                # The forecast method should predict `prediction_length` steps ahead
                forecast_df, _ = self.forecast(train_df) # Pass train_df as context

                # --- Merge and Evaluate ---
                # Ensure forecast_df has the date column named correctly ('ds' typically)
                # and actuals_df has the original date column name.
                # We need to merge on the date. Rename forecast 'ds' if needed.
                if 'ds' not in forecast_df.columns:
                     _logger.error(f"Backtest Error (Group {group_id_val}, Window {window_num}): Forecast DataFrame missing 'ds' column.")
                     continue # Skip window if forecast format is wrong

                # Prepare actuals for merging: select relevant columns and ensure date format matches forecast
                actuals_to_merge = actuals_df[[self.date_col, self.target]].rename(columns={self.date_col: 'ds'})
                actuals_to_merge['ds'] = pd.to_datetime(actuals_to_merge['ds'])

                # Merge actuals and forecasts on 'ds'
                # Use outer merge to see mismatches, left merge might be safer if forecast dates are guaranteed
                merged = pd.merge(actuals_to_merge, forecast_df[['ds', 'y_pred']], on='ds', how='inner')

                if merged.empty:
                    _logger.warning(f"Backtest window {window_num} for group {group_id_val}: Merged actuals and forecast is empty. Check date alignment.")
                    curr_test_start_date += self.stride_offset
                    continue

                # Calculate metric (ensure y_true and y_pred are aligned)
                # Handle potential NaNs if metric function requires it
                y_true = merged[self.target].fillna(0) # Example: fill NaNs in target
                y_pred = merged['y_pred'].fillna(0) # Example: fill NaNs in prediction

                # Use the metric defined in the config
                if self.metric.lower() == "smape":
                    error = smape_loss(y_true, y_pred)
                # Add other metrics here (e.g., mape, rmse)
                # elif self.metric.lower() == "mape":
                #     error = mean_absolute_percentage_error(y_true, y_pred) # Need import
                # elif self.metric.lower() == "rmse":
                #     error = np.sqrt(mean_squared_error(y_true, y_pred)) # Need import
                else:
                    _logger.warning(f"Unsupported metric '{self.metric}' specified. Defaulting to NaN.")
                    error = np.nan

                # Append result
                results.append((group_id_val, model_name, self.metric, error))
                # _logger.debug(f"Backtest window {window_num} for group {group_id_val}: {self.metric}={error:.4f}")

            except Exception as e:
                _logger.error(f"Error during backtest window {window_num} for group {group_id_val}: {e}", exc_info=False) # Set exc_info=True for traceback
                # Append error result
                results.append((group_id_val, model_name, self.metric, float('inf'))) # Indicate error

            # Move to the next window start date based on stride
            curr_test_start_date += self.stride_offset

        if not results:
            _logger.warning(f"Backtesting for group {group_id_val} produced no results. Check data range and parameters.")
        return results

