from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    redis_url: str
    app_name: str = "Whot Online"


settings = Settings()