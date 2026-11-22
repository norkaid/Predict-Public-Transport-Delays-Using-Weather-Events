"""Column names and project-wide constants.

Column names follow the Kaggle dataset
``khushikyad001/public-transport-delays-with-weather-and-events``. Raw column
names are normalised (lower-cased, spaces -> underscores) on load, so minor
spelling differences in the header do not break the pipeline.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "public_transport_delays.csv"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42

# --- Raw columns -----------------------------------------------------------
ID_COL = "trip_id"
DATE_COL = "date"
TIME_COL = "time"
SCHED_DEP_COL = "scheduled_departure"
SCHED_ARR_COL = "scheduled_arrival"

TARGET = "actual_arrival_delay_min"
DEPARTURE_DELAY = "actual_departure_delay_min"
DELAYED_FLAG = "delayed"

WEATHER_COL = "weather_condition"
TEMP_COL = "temperature_c"
HUMIDITY_COL = "humidity_percent"
WIND_COL = "wind_speed_kmh"
PRECIP_COL = "precipitation_mm"
EVENT_COL = "event_type"
ATTENDANCE_COL = "event_attendance_est"
CONGESTION_COL = "traffic_congestion_index"
HOLIDAY_COL = "holiday"
PEAK_COL = "peak_hour"
WEEKDAY_COL = "weekday"
SEASON_COL = "season"

TRANSPORT_COL = "transport_type"
ROUTE_COL = "route_id"
ORIGIN_COL = "origin_station"
DEST_COL = "destination_station"

NUMERIC_RAW = [
    TEMP_COL, HUMIDITY_COL, WIND_COL, PRECIP_COL,
    ATTENDANCE_COL, CONGESTION_COL,
]
BOOL_RAW = [HOLIDAY_COL, PEAK_COL]

# Physically plausible ranges used to clip sensor/entry errors.
VALID_RANGES = {
    TEMP_COL: (-50.0, 55.0),
    HUMIDITY_COL: (0.0, 100.0),
    WIND_COL: (0.0, 200.0),
    PRECIP_COL: (0.0, 300.0),
    ATTENDANCE_COL: (0.0, None),
    CONGESTION_COL: (0.0, None),
}

NO_EVENT_LABELS = {"", "none", "no event", "nan", "null", "unknown"}

# A trip counts as "delayed" when it arrives this many minutes late or more.
DELAY_THRESHOLD_MIN = 5.0

# Columns never used as model inputs. The departure delay is observed at the
# same moment as the arrival delay, so using it would leak the answer.
LEAKY_COLS = [DEPARTURE_DELAY, DELAYED_FLAG]
