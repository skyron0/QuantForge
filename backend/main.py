from fastapi import FastAPI
from configs.settings import settings
from configs.logging import app_logger

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
)

@app.on_event("startup")
async def startup():

    app_logger.info("QuantForge starting...")

    app_logger.info(f"Environment : {settings.ENVIRONMENT}")

    app_logger.info("Logger Ready")


@app.get("/")
def root():

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
def health():

    app_logger.info("/health endpoint called")

    return {
        "status": "healthy"
    }