import logging

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with stream handler and formatter.

    :param name: Name of the logger (usually __name__)
    :return: Configured logging.Utils instance
    """
    logger = logging.getLogger(name)

    # Only add handler if no handlers are attached (avoid duplicates)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()

        # Create and set formatter for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False  # Prevent double logging in some frameworks

    return logger
