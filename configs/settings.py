from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    PROJECT_NAME: str
    VERSION: str
    ENVIRONMENT: str

    PRIMARY_EXCHANGE: str

    SYMBOLS: str

    COLLECTOR_INTERVAL: int

    QUEUE_MAX_SIZE: int

    DATABASE_URL: str

    REDIS_URL: str

    LOG_LEVEL: str

    OLLAMA_URL: str

    BYBIT_WS: str

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()