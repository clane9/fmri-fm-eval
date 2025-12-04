import importlib
import pkgutil
import warnings
from typing import Callable


import fmri_fm_eval.models
from fmri_fm_eval.models.base import ModelTransform, ModelWrapper


_MODEL_REGISTRY: dict[str, Callable[..., ModelWrapper]] = {}
_TRANSFORM_REGISTRY: dict[str, Callable[..., ModelTransform]] = {}


def register_model(name_or_func: str | Callable | None = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _MODEL_REGISTRY:
            warnings.warn(f"Model {name} already registered.", RuntimeWarning)
        _MODEL_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_model(name: str, **kwargs) -> ModelWrapper:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model {name} not registered")
    model = _MODEL_REGISTRY[name](**kwargs)
    return model


def list_models() -> list[str]:
    return list(_MODEL_REGISTRY)


def register_transform(name_or_func: str | Callable | None = None):
    def _decorator(func: Callable):
        name = name_or_func if isinstance(name_or_func, str) else func.__name__
        if name in _TRANSFORM_REGISTRY:
            warnings.warn(f"Transform {name} already registered.", RuntimeWarning)
        _TRANSFORM_REGISTRY[name] = func
        return func

    if isinstance(name_or_func, Callable):
        return _decorator(name_or_func)
    return _decorator


def create_transform(name: str, **kwargs) -> ModelTransform:
    if name not in _TRANSFORM_REGISTRY:
        raise ValueError(f"Transform {name} not registered")
    transform = _TRANSFORM_REGISTRY[name](**kwargs)
    return transform


def list_transforms() -> list[str]:
    return list(_TRANSFORM_REGISTRY)


def import_plugins():
    # https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages
    modules = {
        name: importlib.import_module(name)
        for finder, name, ispkg in pkgutil.iter_modules(
            fmri_fm_eval.models.__path__, fmri_fm_eval.models.__name__ + "."
        )
    }
    return modules
