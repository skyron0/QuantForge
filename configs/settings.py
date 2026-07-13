from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    PROJECT_NAME: str

    VERSION: str

    ENVIRONMENT: str

    OLLAMA_URL: str

    DATABASE_URL: str

    REDIS_URL: str

    LOG_LEVEL: str

    class Config:
        env_file = ".env"


settings = Settings()