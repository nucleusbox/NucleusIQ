"""Root exception for the NucleusIQ framework.

All subsystem error modules import from this file to avoid circular
imports with the ``errors/__init__.py`` re-export layer.
"""

from __future__ import annotations


class NucleusIQError(Exception):
    """Base exception for all NucleusIQ framework errors."""
