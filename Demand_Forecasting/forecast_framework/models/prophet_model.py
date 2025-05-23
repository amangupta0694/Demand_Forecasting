# Demand_Forecasting/forecast_framework/models/prophet_model.py
# Wrapper for the Prophet forecasting model.

import pandas as pd
from prophet import Prophet
import logging
from omegaconf import DictConfig # Import DictConfig

# Import the abstract base class
from forecast_framework.abstract_model import ForecastingRegressor

_logger = logging.getLogger(__name__)

class SKTimeProphet(ForecastingRegressor):
    """
    Wrapper class for Facebook Prophet model, conforming to the ForecastingRegressor interface.
    """
    def __init__(self, config: DictConfig):
        """
        Initializes the SKTimeProphet model.

        Args:
            config: The merged OmegaConf DictConfig object containing both
                    forecasting and model configurations. Expected structure:
                    {
                        'forecasting': {...},
                        'models': { 'SKTimeProphet': {...}, ... }
                    }
        """
        # Pass the entire config to the parent class.
        # The parent extracts general forecasting params like 'freq', 'prediction_length'.
        super().__init__(config) # Pass the whole config

        # Extract Prophet-specific parameters from the 'models.SKTimeProphet' section
        try:
            model_params = config.models.SKTimeProphet
            _logger.info(f"Found SKTimeProphet parameters: {model_params}")
        except Exception as e:
             _logger.error(f"Could not find or access parameters under 'models.SKTimeProphet' in the config: {e}")
             # Provide default values or raise an error if params are essential
             model_params = {} # Example: Use empty dict, Prophet will use its defaults

        # Initialize the Prophet model using extracted parameters
        # Use .get() with defaults for robustness
        self.model = Prophet(
            changepoint_prior_scale=model_params.get('changepoint_prior_scale', 0.05),
            seasonality_mode=model_params.get('seasonality_mode', 'additive'), # Changed default to additive as it's Prophet's default
            holidays=model_params.get('holidays', None), # Handle holidays later if needed (e.g., load from path)
            growth=model_params.get('growth', 'linear'),
            daily_seasonality=model_params.get('daily_seasonality', 'auto'), # Use Prophet's auto detection
            weekly_seasonality=model_params.get('weekly_seasonality', 'auto'),
            yearly_seasonality=model_params.get('yearly_seasonality', 'auto'),
            interval_width=model_params.get('interval_width', 0.80) # Prophet's default
            # Add other Prophet parameters here if needed, accessing them from model_params
        )
        _logger.info(f"Initialized Prophet model with specific params: {model_params}")
        # Note: General params like 'freq', 'prediction_length' are available via self.params.forecasting

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for Prophet: expects columns 'ds' and 'y'.
        Uses 'date_col' and 'target' from the forecasting config.
        """
        prep_df = df.copy()
        date_col = self.params.forecasting.date_col # Access from forecasting section
        target_col = self.params.forecasting.target   # Access from forecasting section

        # Ensure date column is datetime
        prep_df[date_col] = pd.to_datetime(prep_df[date_col])
        # Rename columns to 'ds' and 'y'
        prep_df = prep_df.rename(columns={
            date_col: 'ds',
            target_col: 'y'
        })
        # Select only necessary columns (and potentially group_id if needed later)
        required_cols = ['ds', 'y']
        # Add 'cap' if growth is logistic - requires config check
        if self.model.growth == 'logistic':
             if 'cap' not in prep_df.columns:
                 _logger.error("Logistic growth requires a 'cap' column in the data.")
                 raise ValueError("Missing 'cap' column for logistic growth.")
             required_cols.append('cap')
             if 'floor' in prep_df.columns: # Optional floor
                 required_cols.append('floor')

        return prep_df[required_cols]

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the Prophet model.
        """
        group_id_col = self.params.forecasting.group_id # Access from forecasting section
        group_id_val = train_df[group_id_col].iloc[0] if group_id_col in train_df.columns else 'N/A'
        _logger.info(f"Fitting Prophet model for group: {group_id_val}")

        prepared_train_df = self.prepare_data(train_df)
        try:
            self.model.fit(prepared_train_df)
            _logger.info(f"Prophet model fitting complete for group: {group_id_val}.")
        except Exception as e:
             _logger.error(f"Error fitting Prophet model for group {group_id_val}: {e}")
             # Re-raise or handle as appropriate
             raise e
        return self # Return self for chaining

    def forecast(self, df: pd.DataFrame) -> (pd.DataFrame, object):
        """
        Generate forecasts using the fitted Prophet model.

        Args:
            df: The historical data (Pandas DataFrame) needed by Prophet to create
                the future dataframe. It should contain the history for the group
                being forecasted.

        Returns:
            A tuple containing:
            - forecast_df (pd.DataFrame): DataFrame with 'ds', 'y_pred', and potentially 'unique_id'.
            - model_instance (Prophet): The fitted Prophet model instance.
        """
        group_id_col = self.params.forecasting.group_id
        group_id_val = df[group_id_col].iloc[0] if group_id_col in df.columns else 'N/A'
        _logger.info(f"Generating forecast with Prophet for group: {group_id_val}")

        # Prediction length and frequency from forecasting config
        prediction_length = self.params.forecasting.prediction_length
        freq = self.params.forecasting.freq

        try:
            # Create future dataframe for prediction
            # Prophet's fit needs to have run before make_future_dataframe
            future_df = self.model.make_future_dataframe(
                periods=prediction_length,
                freq=freq # Use freq from main config
            )
            # Add cap/floor to future_df if using logistic growth
            if self.model.growth == 'logistic':
                 # Need to define how cap/floor are projected into the future.
                 # Often, they are constant or based on some logic.
                 # Assuming they are present in the historical `df` and we can forward-fill or use the last known value.
                 # This requires the original `df` (before prepare_data) or careful handling in prepare_data.
                 # For simplicity, let's assume 'cap' (and 'floor') needs to be added manually here if not in history.
                 # This is a common challenge with Prophet's logistic growth.
                 # Placeholder: Use last known cap/floor from history.
                 if 'cap' in df.columns:
                     future_df['cap'] = df['cap'].iloc[-1] # Simplistic: use last value
                 else:
                      _logger.error(f"Logistic growth selected, but 'cap' column missing in historical data for group {group_id_val}.")
                      raise ValueError("Missing 'cap' column for logistic growth forecast.")
                 if 'floor' in df.columns: # Optional floor
                     future_df['floor'] = df['floor'].iloc[-1] # Simplistic: use last value


            # Predict
            forecast_result = self.model.predict(future_df)
            _logger.info(f"Prophet forecast generation complete for group: {group_id_val}.")

            # --- Output Formatting ---
            # Select relevant columns: ds, yhat (forecast), potentially others like yhat_lower, yhat_upper
            # Rename 'yhat' to 'y_pred' for consistency with the framework.
            forecast_output = forecast_result[['ds', 'yhat']].rename(columns={'yhat': 'y_pred'})

            # Filter to only include the actual forecast period (optional, depends on downstream use)
            # last_history_date = df[self.params.forecasting.date_col].max() # Use original date col name from df
            # last_history_date_dt = pd.to_datetime(last_history_date)
            # forecast_output = forecast_output[forecast_output['ds'] > last_history_date_dt].copy()

            # Add back the group_id column, renaming it to 'unique_id' as expected by scoring UDF schema
            if group_id_col in df.columns:
                 forecast_output['unique_id'] = df[group_id_col].iloc[0]
            else:
                 # This case should ideally not happen if data is grouped correctly upstream
                 _logger.warning(f"Group ID '{group_id_col}' not found in input dataframe during forecast for (potential) group {group_id_val}.")
                 forecast_output['unique_id'] = "unknown_group" # Placeholder

            # Ensure columns match the expected scoring schema: unique_id, ds, y_pred
            # Reorder if necessary
            forecast_output = forecast_output[['unique_id', 'ds', 'y_pred']]

            # Convert 'ds' back to DateType if it became Timestamp
            # forecast_output['ds'] = forecast_output['ds'].dt.date # Keep as datetime for potential time components

            return forecast_output, self.model

        except Exception as e:
            _logger.error(f"Error generating forecast with Prophet for group {group_id_val}: {e}")
            # Return empty dataframe matching schema or handle as appropriate
            return pd.DataFrame(columns=['unique_id', 'ds', 'y_pred']), self.model

