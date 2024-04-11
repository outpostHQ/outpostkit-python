from outpostkit._types.repository import REPOSITORY_TYPES
from outpostkit.client import Client
from outpostkit.resource import Namespace


# Assuming path always starts with `/`, can create a parser for this (src/... -> /src/...)
class Repository(Namespace):
    def __init__(
        self, client: Client, repo_type: REPOSITORY_TYPES, entity: str, name: str
    ) -> None:
        self.entity = entity
        self.name = name
        self.repo_type = repo_type
        self.fullName = f"{entity}/{name}"
        super().__init__(client)

    def view_blob(self, path: str, ref: str = "HEAD", raw: bool = True):
        resp = self._client._request(
            path=f"/git/blobs/{self.repo_type}/{self.fullName}/view/{ref}{path}",
            method="GET",
            params={"raw": raw},
        )
        resp.raise_for_status()

        return resp.json()  # TODO Type

    def download_blob(self, path: str, ref: str = "HEAD", raw: bool = True):
        resp = self._client._request(
            path=f"/git/blobs/{self.repo_type}/{self.fullName}/download/{ref}{path}",
            method="GET",
            params={"raw": raw},
            stream=True,
        )
        resp.raise_for_status()

        return resp.json()  # TODO Type

    def view_tree(
        self,
        ref: str = "HEAD",
        path: str = "/",
        with_commit=False,
        with_metadata=False,
    ):
        resp = self._client._request(
            path=f"/git/tree/{self.repo_type}/{self.fullName}/view/{ref}{path}",
            method="GET",
            params={"with_commit": with_commit, "with_metadata": with_metadata},
        )
        resp.raise_for_status()

        return resp.json()  # TODO Type

    def search_tree(
        self,
        search: str,
        ref: str = "HEAD",
    ):
        resp = self._client._request(
            path=f"/git/tree/{self.repo_type}/{self.fullName}/search",
            method="GET",
            params={"search": search, "ref": ref},
        )
        resp.raise_for_status()

        return resp.json()  # TODO Type


class RepositoryAtRef(Namespace):
    def __init__(
        self,
        client: Client,
        repo_type: REPOSITORY_TYPES,
        entity: str,
        name: str,
        ref: str,
    ) -> None:
        self.repo = Repository(
            client=client, repo_type=repo_type, entity=entity, name=name
        )
        self.ref = ref
        super().__init__(client)

    def view_blob(self, path: str, raw: bool = True):
        return self.repo.view_blob(path=path, ref=self.ref, raw=raw)

    def download_blob(self, path: str, raw: bool = True):
        return self.repo.download_blob(path=path, ref=self.ref, raw=raw)

    def view_tree(
        self,
        path: str = "/",
        with_commit=False,
        with_metadata=False,
    ):
        return self.repo.view_tree(
            path=path,
            ref=self.ref,
            with_commit=with_commit,
            with_metadata=with_metadata,
        )

    def search_tree(
        self,
        search: str,
    ):
        return self.repo.search_tree(search=search, ref=self.ref)
