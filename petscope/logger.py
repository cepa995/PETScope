import logging
import warnings

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logging.getLogger("nibabel").setLevel(logging.WARNING)
logger = logging.getLogger() 
