from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EntityStats:
    followers_count: int
    following_count: int


@dataclass
class UserDetails:
    id: str
    createdAt: str
    updatedAt:str
    stats: EntityStats
    avatarUrl: str
    name: str
    bio: Optional[str]
    socials: Optional[Dict[str,str]]
    displayName:str

@dataclass
class UserShortDetails:
    id:str
    name: str
    avatarUrl: str
