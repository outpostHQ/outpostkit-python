from dataclasses import dataclass
from typing import Literal, Optional

ENTITY_TYPES = Literal["user", "team"]


@dataclass
class FollowEntity:
    name: str
    id: str
    type: ENTITY_TYPES
    avatarUrl: str
    isFollowing: Optional[bool]
