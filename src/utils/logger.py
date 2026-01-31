
import os

import logging
import logging.handlers
from typing import Optional

def configure_logger(console_level: int, file_level: int, path: Optional[str] = None) -> bool:
    if path is None:
        path = os.environ.get(
            "APP_LOG_PATH",
            os.path.join(os.getcwd(), "logs", "app.log"),
        )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(console_level)
    path_dir = os.path.dirname(path)
    os.makedirs(path_dir, exist_ok=True)
    file = logging.handlers.TimedRotatingFileHandler(path, when="midnight", backupCount=7)
    file.setLevel(file_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)
    return True
