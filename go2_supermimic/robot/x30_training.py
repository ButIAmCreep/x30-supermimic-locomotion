"""Compatibility import; implementation: robots.x30.robot."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("go2_supermimic.robots.x30.robot")
_sys.modules[__name__] = _module
