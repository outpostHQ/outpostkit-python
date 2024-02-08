from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class UserStats:
    followers_count: int
    following_count: int


@dataclass
class UserDetails:
    id: str
    createdAt: str
    updatedAt: str
    stats: UserStats
    avatarUrl: str
    name: str
    bio: Optional[str]
    socials: Optional[Dict[str, str]]
    displayName: str

    def __init__(self, *args, **kwargs) -> None:
        for field in self.__annotations__:
            if field == "stats":
                self.stats = UserStats(**kwargs.get("stats"))
            else:
                setattr(self, field, kwargs.get(field))


@dataclass
class UserShortDetails:
    id: str
    name: str
    avatarUrl: str
