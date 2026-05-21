"""Utilities package bootstrap for local submodules."""

import os
import sys

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_SUBMODULES_MAST3R_ROOT = os.path.join(_UTILS_DIR, 'submodules', 'mast3r')
_SUBMODULES_DUST3R_INNER = os.path.join(_SUBMODULES_MAST3R_ROOT, 'dust3r')

# add root containing 'mast3r' and 'dust3r' directories to sys.path
if os.path.isdir(_SUBMODULES_MAST3R_ROOT) and _SUBMODULES_MAST3R_ROOT not in sys.path:
    sys.path.insert(0, _SUBMODULES_MAST3R_ROOT)
if os.path.isdir(_SUBMODULES_DUST3R_INNER) and _SUBMODULES_DUST3R_INNER not in sys.path:
    sys.path.insert(0, _SUBMODULES_DUST3R_INNER)