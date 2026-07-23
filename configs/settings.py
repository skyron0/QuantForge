from typing import Optional
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

    # AI Runtime Configuration
    AI_PROVIDER: str = "ollama"
    AI_MODEL: Optional[str] = None
    AI_BASE_URL: str = "http://localhost:11434"
    AI_CONNECTION_TIMEOUT_SECONDS: float = 5.0
    AI_INFERENCE_TIMEOUT_SECONDS: float = 30.0
    AI_STRUCTURED_MAX_RETRIES: int = 3
    AI_CONTEXT_TTL_SECONDS: float = 300.0

    # Persistence Configuration
    PERSISTENCE_ENABLED: bool = True
    PERSISTENCE_BACKEND: str = "postgres"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_POOL_MAX_OVERFLOW: int = 10
    DATABASE_CONNECT_TIMEOUT_SECONDS: float = 5.0

    # Feature Runtime Configuration
    FEATURE_RUNTIME_ENABLED: bool = True
    FEATURE_MINIMUM_HISTORY: int = 100
    FEATURE_STALENESS_LIMIT_SECONDS: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()