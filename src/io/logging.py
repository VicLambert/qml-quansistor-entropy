
import logging


def configure_logger(console_level, file_level, path="qml-quansistor-entropy/logs/app.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(console_level)

    file = logging.handlers.TimedRotatingFileHandler(path, when="midnight", backupCount=7)
    file.setLevel(file_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    file.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file)
    return True
