from typing import Dict, Type


class Registry:
    """General purpose module registry"""

    def __init__(self, name: str):
        """

        :param name: registry name
        """
        self.name = name
        self._module: Dict[str, Type] = {}

    def get(self, key, kwargs):
        """
        :raises TypeError kwargs must be a dict
        :raises KeyError on not registered
        """
        module = self._module.get(key)
        if not isinstance(kwargs, dict):
            raise TypeError('kwargs must be a dict!')
        if module:
            return module(**kwargs)
        else:
            raise KeyError(f'Module {key} is not registered!')

    def register(self, cls):
        """
        :raises ValueError already registered
        """
        module_name = cls.__name__
        if module_name in self._module:
            raise ValueError(f'Module {module_name} already registered!')
        else:
            self._module[module_name] = cls

        return cls
