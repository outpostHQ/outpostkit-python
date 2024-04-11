from typing import List, Optional

from outpostkit._types.finetuning import FinetuningServiceCreateResponse
from outpostkit._utils.constants import OutpostSecret
from outpostkit.client import Client
from outpostkit.resource import Namespace


class FinetuningService(Namespace):
    def __init__(self, client: Client, entity: str, name: str) -> None:
        self.entity = entity
        self.name = name
        self.fullName = f"{entity}/{name}"
        super().__init__(client)


class Finetunings(Namespace):
    def __init__(self, client: Client, entity: str, name: str) -> None:
        self.entity = entity
        self.name = name
        self.fullName = f"{entity}/{name}"
        super().__init__(client)

    def list(self):
        resp = self._client._request("GET", f"/finetunings/{self.entity}")
        return resp.json()

    def create(
        self,
        name: str,
        task_type: str,
        dataset: str,
        train_path: str,
        validation_path: str,
        secrets: Optional[List[OutpostSecret]] = None,
    ) -> FinetuningService:
        resp = self._client._request("POST", f"/finetunings/{self.entity}")
        obj = FinetuningServiceCreateResponse(**resp.json())
        return FinetuningService(client=self._client, entity=self.entity, name=obj.name)
