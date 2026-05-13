import logging


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("yfcc")


log = logging.getLogger("yfcc")
