from typing import List, Optional

from outpostkit._types.team import TeamDetails
from outpostkit._types.user import UserDetails
from outpostkit.client import Client

from ._types.entity import FollowEntity
from .resource import Namespace


class Team(Namespace):
    def __init__(
        self,
        client: Client,
        name: str,
    ) -> None:
        self.name = name
        super().__init__(client)

    def get(self) -> TeamDetails:
        """Get Team

        Returns:
            The Team details.
        """
        resp = self._client._request(path=f"/teams/{self.name}", method="GET")
        resp.raise_for_status()

        return TeamDetails(**resp.json())

    async def async_get(self) -> TeamDetails:
        """Get Team

        Returns:
            The Team details.
        """
        resp = await self._client._async_request(
            path=f"/teams/{self.name}", method="GET"
        )
        resp.raise_for_status()

        return TeamDetails(**resp.json())

    def list_followers(
        self,
        include_is_following: bool = False,
    ) -> List[FollowEntity]:
        """List followers"""
        resp = self._client._request(
            path=f"/teams/{self.name}/followers",
            params={"isFollowing": include_is_following},
            method="GET",
        )
        resp.raise_for_status()

        return [FollowEntity(**data) for data in resp.json()]

    def list_following(
        self,
        include_is_following: bool = False,
    ) -> List[FollowEntity]:
        """List following"""
        resp = self._client._request(
            path=f"/teams/{self.name}/following",
            params={"isFollowing": include_is_following},
            method="GET",
        )
        resp.raise_for_status()

        return [FollowEntity(**data) for data in resp.json()]

    def upload_avatar(self, avatar_path: str) -> int:
        """Delete avatar"""
        resp = self._client._request(
            path=f"/teams/{self.name}/avatar",
            method="POST",
            files={"avatar": open(avatar_path)},
        )
        resp.raise_for_status()

        return resp.status_code

    def delete_avatar(self) -> int:
        """Follow avatar"""
        resp = self._client._request(path=f"/teams/{self.name}/avatar", method="DELETE")
        resp.raise_for_status()

        return resp.status_code

    def update_socials(
        self,
        website: Optional[str],
        x: Optional[str],
    ) -> int:
        """Follow avatar"""
        resp = self._client._request(
            path=f"/teams/{self.name}/avatar",
            method="PUT",
            json={"website": website, "x": x},  # TODO fix null vs undefined
        )
        resp.raise_for_status()

        return resp.status_code

    def delete(self) -> int:
        """Delete team"""
        resp = self._client._request(
            path=f"/teams/{self.name}",
            method="DELETE",
        )
        resp.raise_for_status()

        return resp.status_code

    def members(self) -> List[UserDetails]:
        """List Team members. xx under development xx"""
        resp = self._client._request(
            path=f"/teams/{self.name}",
            method="GET",
        )
        resp.raise_for_status()

        return [UserDetails(**team) for team in resp.json()]
