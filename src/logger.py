import logging
import os


def configure_logging(level: str = "INFO") -> None:
    """Configure logging with the specified level.

    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.storage").setLevel(logging.WARNING)


configure_logging(os.getenv("LOGGING_LEVEL", "INFO"))

# Default logger instance
logger = logging.getLogger("pipeline")

