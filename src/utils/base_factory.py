from abc import ABC
from pathlib import Path
from typing import Callable

from loguru import logger

from ..utils.load_modules import load_modules
from ..utils.utils import get_library_root


def initialize_package_factory(init_file_path: str) -> None:
    """
    Initialize a module factory.

    This function is used to initialize a module factory. It loads all the
    modules in the directory of the ``init_file_path`` so that they are registered
    in the module factory.

    :param init_file_path: Path to the ``__init__.py`` file.
    """
    init_file_path = Path(init_file_path).resolve()
    module_path = init_file_path.parent.relative_to(get_library_root()).parts
    module_path = ".".join(module_path) + ".{}"
    load_modules(
        module_files=map(str, init_file_path.parent.glob("*.py")),
        module_path=module_path,
    )


class BaseFactory:
    """Base factory class."""

    base_class = ABC
    registry = {}

    @classmethod
    def register(cls) -> Callable[[base_class], base_class]:
        """
        Register classes to the internal registry.

        :return: Callable to get the object.
        """

        def inner_wrapper(wrapped_class: cls.base_class) -> cls.base_class:
            name = wrapped_class.__name__
            if name in cls.registry:
                logger.warning(
                    f"{cls.base_class.__name__} {name} already exists, will replace it."
                )
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, class_name: str, **kwargs) -> base_class:
        """
        Create a class instance.

        This method gets the appropriate class from the registry
        and creates an instance of it, while passing in the parameters
        given in ``kwargs``.

        :param class_name: Name of the class to create.
        :param kwargs: Parameters to instantiate the class instance.
        :return: Instance of the requested class.
        :raises KeyError: If the class name is not in the registry.
        """
        if class_name not in cls.registry:
            raise KeyError(
                f"{cls.base_class.__name__} {class_name} is not in the registry."
            )
        class_ = cls.registry[class_name]
        if hasattr(class_, "from_config"):
            instance = class_.from_config(kwargs)
        else:
            instance = class_(**kwargs)
        return instance
