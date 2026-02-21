"""Backward-compatible shim for the T-construction extension module.

The binary module now lives at ``cripser.tcripser`` so both compiled modules
are placed under the same package directory.
"""

from cripser.tcripser import *  # noqa: F401,F403
