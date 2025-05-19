import logging


def get_logger():
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    return logger
