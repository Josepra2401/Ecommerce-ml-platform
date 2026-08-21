# E-Commerce ML Platform — Modules Package
from .data_loader import load_raw, guess_columns
from .forecasting import prepare_for_forecast, fit_prophet_model, run_forecast_tab
from .segmentation import prepare_for_rfm, segment_rfm, run_rfm_tab
from .geo_analysis import try_geocode_cities, run_geo_tab
