from __future__ import annotations

from typing import Optional

from torch import Tensor


class SelfState(object):
    """Lightweight container for the current EFES self state."""

    __slots__ = (
        "z_self",
        "agency",
        "phase",
        "continuity",
        "rupture_memory",
        "valid_mask",
    )

    def __init__(
        self,
        z_self: Tensor,
        agency: Tensor,
        phase: Tensor,
        continuity: Tensor,
        rupture_memory: Tensor,
        valid_mask: Tensor,
    ) -> None:
        self.z_self = z_self
        self.agency = agency
        self.phase = phase
        self.continuity = continuity
        self.rupture_memory = rupture_memory
        self.valid_mask = valid_mask

    def as_dict(self, prefix: Optional[str] = None):
        if prefix is None:
            prefix = ""
        return {
            prefix + "z_self": self.z_self,
            prefix + "agency": self.agency,
            prefix + "phase": self.phase,
            prefix + "continuity": self.continuity,
            prefix + "rupture_memory": self.rupture_memory,
            prefix + "valid_mask": self.valid_mask,
        }
