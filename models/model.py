"""Legacy ResNet model compatibility shim.

Active uSERS-Net code lives under ``sers.models.usersnet`` and
``models.stacking_utils``. This module remains only for old scripts that still
import ``models.model``.
"""

from models.legacy.model import *  # noqa: F401,F403
