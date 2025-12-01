from loguru import logger

from ..utils.base_factory import initialize_package_factory

logger.debug("Initializing Alignment Layers")
initialize_package_factory(__file__)
