from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    redis_url: str
    app_name: str = "Whot Online"
    guest_session_cookie_name: str = 'player_id'
    guest_session_max_age_days: int = 30
    gemini_api_key: str = ""
    game_ttl_seconds: int = 86400
    completed_game_ttl_seconds: int = 3600
    game_lock_timeout_seconds: int = 10


settings = Settings()