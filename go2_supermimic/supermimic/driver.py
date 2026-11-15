"""Compatibility import; implementation: robots.go2.driver."""

from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("go2_supermimic.robots.go2.driver")
_sys.modules[__name__] = _module
