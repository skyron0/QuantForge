from loguru import logger
import sys

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
           "<level>{message}</level>",
)

logger.add(
    "logs/system/quantforge.log",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    level="INFO",
)

app_logger = logger