import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ODDSPAPI_API_KEY = os.getenv("ODDSPAPI_API_KEY", "")
PANDASCORE_API_KEY = os.getenv("PANDASCORE_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
BETFAIR_APP_KEY = os.getenv("BETFAIR_APP_KEY", "")
BETFAIR_USERNAME = os.getenv("BETFAIR_USERNAME", "")
BETFAIR_PASSWORD = os.getenv("BETFAIR_PASSWORD", "")

# Database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cs2.db")

# Scraping
HLTV_SCRAPE_INTERVAL_HOURS = 6
MIN_MATCHES_FOR_PREDICTION = 3

# Betting
MIN_EDGE_PERCENT = 5.0  # minimum edge to place a bet
KELLY_FRACTION = 0.25   # quarter Kelly for safety
MAX_BET_PERCENT = 3.0   # max % of bankroll per bet
INITIAL_BANKROLL = 1000  # USD
