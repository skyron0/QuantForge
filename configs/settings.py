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

    TAKE_PROFIT: float = 100.0
    STOP_LOSS: float = 50.0
    BUY_THRESHOLD: int = 40
    SELL_THRESHOLD: int = -40
    MIN_CANDLES: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()