import logging


def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    return logger
