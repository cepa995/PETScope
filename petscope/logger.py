import logging

# Configure Basic Logging
# ------------------------
# The logging system is initialized with a basic configuration that specifies:
# - Log message format: Includes timestamp, log level, and message.
# - Default log level: INFO (logs all messages at INFO level and above).
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", 
    level=logging.INFO
)

# Adjust Logging Levels for Specific Libraries
# --------------------------------------------
# The logging level for the "nibabel" library is set to WARNING. This suppresses
# INFO and DEBUG messages from nibabel, which can clutter the output during PET image processing.
logging.getLogger("nibabel").setLevel(logging.WARNING)

# Define Global Logger
# ---------------------
# The logger object is initialized as the default logger for use across the framework.
# This logger can be imported and used in other modules to log messages consistently.
logger = logging.getLogger()

# Example Usage
# -------------
# To use this logger in another module:
#     from petscope.logger import logger
#     logger.info("This is an informational message.")
#     logger.warning("This is a warning message.")
