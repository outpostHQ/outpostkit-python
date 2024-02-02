from typing import Dict, Optional

from pydantic import BaseModel


class EntityStats(BaseModel):
    followers_count: int
    following_count: int


class UserDetails(BaseModel):
    id: str
    createdAt: str
    updatedAt:str
    stats: EntityStats
    avatarUrl: str
    name: str
    bio: Optional[str]
    socials: Optional[Dict[str,str]]
    displayName:str
