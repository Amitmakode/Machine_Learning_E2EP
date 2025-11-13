# package marker
# laptop_price/__init__.py
from laptop_price.logger import get_logger

logger = get_logger(__name__)
logger.info(" Laptop Price Prediction package initialized")
