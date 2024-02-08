from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TeamStats:
    followers_count: int


@dataclass
class TeamDetails:
    id: str
    createdAt: str
    updatedAt: str
    stats: TeamStats
    avatarUrl: str
    name: str
    bio: Optional[str]
    socials: Optional[Dict[str, str]]
    displayName: str

    def __init__(self, *args, **kwargs) -> None:
        for field in self.__annotations__:
            if field == "stats":
                self.stats = TeamStats(**kwargs.get("stats"))
            else:
                setattr(self, field, kwargs.get(field))
