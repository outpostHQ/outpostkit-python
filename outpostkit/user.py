from typing import List, Optional

from outpostkit._types.team import TeamDetails
from outpostkit._types.user import UserDetails
from outpostkit.client import Client

from ._types.entity import FollowEntity
from .resource import Namespace


class User(Namespace):
    def __init__(
        self,
        client: Client,
    ) -> None:
        super().__init__(client)

    def get(self) -> UserDetails:
        """Get User

        Returns:
            The User details.
        """
        resp = self._client._request(path="/user", method="GET")
        resp.raise_for_status()

        return UserDetails(**resp.json())

    async def async_get(self) -> UserDetails:
        """Get User

        Returns:
            The User details.
        """
        resp = await self._client._async_request(path="/user", method="GET")
        resp.raise_for_status()

        return UserDetails(**resp.json())

    def list_followers(self, include_is_following: bool = False) -> List[FollowEntity]:
        """List followers"""
        resp = self._client._request(
            path="/user/followers",
            params={"isFollowing": include_is_following},
            method="GET",
        )
        resp.raise_for_status()

        return [FollowEntity(**data) for data in resp.json()]

    def list_following(
        self,
    ) -> List[FollowEntity]:
        """List following"""
        resp = self._client._request(
            path="/user/following",
            method="GET",
        )
        resp.raise_for_status()

        return [FollowEntity(**data) for data in resp.json()]

    def follow(self, name: str) -> int:
        """Follow entity"""
        resp = self._client._request(
            path=f"/user/following/{name}",
            method="PUT",
        )
        resp.raise_for_status()

        return resp.status_code

    def is_following(self, name: str) -> bool:
        """Does user follow entity"""
        resp = self._client._request(
            path=f"/user/following/{name}",
            method="GET",
        )
        resp.raise_for_status()

        return resp.json()

    def teams(self) -> List[TeamDetails]:
        """List User Teams"""
        resp = self._client._request(
            path="/user/teams",
            method="GET",
        )
        resp.raise_for_status()

        return [TeamDetails(**team) for team in resp.json()]

    def unfollow(self, name: str) -> int:
        """Unfollow entity"""
        resp = self._client._request(
            path=f"/user/following/{name}",
            method="DELETE",
        )
        resp.raise_for_status()

        return resp.status_code

    def upload_avatar(self, avatar_path: str) -> int:
        """Delete avatar"""
        resp = self._client._request(
            path="/user/avatar", method="POST", files={"avatar": open(avatar_path)}
        )
        resp.raise_for_status()

        return resp.status_code

    def delete_avatar(self) -> int:
        """Follow avatar"""
        resp = self._client._request(path="/user/avatar", method="DELETE")
        resp.raise_for_status()

        return resp.status_code

    def update_socials(
        self,
        website: Optional[str],
        x: Optional[str],
    ) -> int:
        """Follow avatar"""
        resp = self._client._request(
            path="/user/avatar",
            method="PUT",
            json={"website": website, "x": x},  # TODO fix null vs undefined
        )
        resp.raise_for_status()

        return resp.status_code

    def delete(self) -> int:
        """Delete user"""
        resp = self._client._request(
            path="/user",
            method="DELETE",
        )
        resp.raise_for_status()

        return resp.status_code
