from dataclasses import asdict
from typing import Any, Dict, List, Literal, Optional, Union

from outpostkit._types.finetuning import (
    FinetuningHFSourceModel,
    FinetuningJobCreationResponse,
    FinetuningJobLog,
    FinetuningModelRepo,
    FinetuningOutpostSourceModel,
    FinetuningResource,
    FinetuningServiceCreateResponse,
    FinetuningsListResponse,
)
from outpostkit._utils.constants import OutpostSecret
from outpostkit._utils.finetuning import FinetuningTask
from outpostkit.client import Client
from outpostkit.resource import Namespace
from outpostkit.utils import parse_finetuning_job_log_data


class FinetuningJob(Namespace):
    def __init__(self, client: Client, entity: str, name: str, job_id: str) -> None:
        self.entity = entity
        self.name = name
        self.fullName = f"{entity}/{name}"
        self.id = job_id
        self._route_prefix = f"/finetunings/{self.fullName}/jobs/{self.id}"
        super().__init__(client)

    def enqueue(self):
        resp = self._client._request(
            "POST", f"/finetunings/{self.entity}/jobs/enqueue", json={"jobs": [self.id]}
        )
        return resp

    def info(
        self,
        with_config: Optional[bool] = None,
        with_trainer_log: Optional[bool] = None,
    ):
        resp = self._client._request(
            "GET",
            self._route_prefix,
            params={
                "cfg": with_config,
                "trainer_log": with_trainer_log,
            },
        )
        return resp

    def configs(self):
        resp = self._client._request(
            "GET",
            f"{self._route_prefix}/configs",
        )
        return resp

    def trainer_logs(self):
        resp = self._client._request(
            "GET",
            f"{self._route_prefix}/logs/trainer",
        )
        return resp

    def delete(self):
        resp = self._client._request("DELETE", self._route_prefix)
        return resp

    def get_logs(
        self,
        log_type: Optional[Literal["dep", "runtime", "event"]] = None,
        start: Optional[Union[int, str]] = None,
        end: Optional[Union[int, str]] = None,
        limit: Optional[int] = 1000,
    ) -> List[FinetuningJobLog]:
        """
        Retrieve logs related to the finetuning job
        Available log types:runtime, dep (deployment) and event.
        Note: the start time defaults to 15 mins ago
        """
        resp = self._client._request(
            "GET",
            f"{self._route_prefix}/logs",
            params={
                "logType": log_type,
                "limit": limit,
                "start": start,
                "end": end,
            },
        )

        return [
            FinetuningJobLog(
                timestamp=str(log.get("timestamp")),
                data=parse_finetuning_job_log_data(log.get("data")),
            )
            for log in resp.json()
        ]


class FinetuningService(Namespace):
    def __init__(self, client: Client, entity: str, name: str) -> None:
        self.entity = entity
        self.name = name
        self.fullName = f"{entity}/{name}"
        self._route_prefix = f"/finetunings/{self.fullName}"
        super().__init__(client)

    def info(self):
        resp = self._client._request("GET", f"{self._route_prefix}")
        return FinetuningResource(**resp.json())

    def list_jobs(
        self,
        status_in: Optional[List[str]] = None,
        status_not_in: Optional[List[str]] = None,
        with_config: Optional[bool] = None,
        with_trainer_log: Optional[bool] = None,
    ):
        resp = self._client._request(
            "GET",
            f"{self._route_prefix}/jobs",
            params={
                "statusIn": ",".join(status_in) if status_in else None,
                "statusNotIn": ",".join(status_not_in) if status_not_in else None,
                "cfg": with_config,
                "trainer_log": with_trainer_log,
            },
        )
        return resp.json()

    def create_job(
        self,
        hardware_instance: str,
        finetuned_model_repo: FinetuningModelRepo,
        configs: Dict[str, Any],
        column_configs: Optional[Dict[str, str]] = None,
        model_source: Literal["huggingface", "outpost", "none"] = "none",
        source_model: Optional[
            Union[FinetuningHFSourceModel, FinetuningOutpostSourceModel]
        ] = None,
        dataset_revision: Optional[str] = "HEAD",
        enqueue: Optional[bool] = None,
    ) -> FinetuningJob:
        resp = self._client._request(
            "POST",
            f"{self._route_prefix}/jobs",
            json={
                "hardwareInstanceId": hardware_instance,
                "configs": configs,
                "columnConfigs": column_configs,
                "modelSource": model_source,
                "sourceHuggingfaceModel": asdict(source_model)
                if isinstance(source_model, FinetuningHFSourceModel)
                else None,
                "sourceOutpostModel": asdict(source_model)
                if isinstance(source_model, FinetuningOutpostSourceModel)
                else None,
                "finetunedModel": asdict(finetuned_model_repo),
                "datasetCommitHash": dataset_revision,
            },
            params={"enqueue": enqueue},
        )
        job_resp = FinetuningJobCreationResponse(**resp.json())
        return FinetuningJob(
            client=self._client, entity=self.entity, name=self.name, job_id=job_resp.id
        )


class Finetunings(Namespace):
    def __init__(self, client: Client, entity: str) -> None:
        self.entity = entity
        self._route_prefix = f"/finetunings/{self.entity}"
        super().__init__(client)

    def list(self):
        resp = self._client._request("GET", self._route_prefix)
        return FinetuningsListResponse(**resp.json())

    def create(
        self,
        name: str,
        task_type: FinetuningTask,
        dataset: str,
        train_path: str,
        validation_path: str,
        secrets: Optional[List[OutpostSecret]] = None,
    ) -> FinetuningService:
        resp = self._client._request(
            "POST",
            self._route_prefix,
            json={
                "name": name,
                "taskType": task_type.value,
                "dataset": dataset,
                "trainPath": train_path,
                "validPath": validation_path,
                "secrets": [asdict(secret) for secret in secrets] if secrets else None,
            },
        )
        obj = FinetuningServiceCreateResponse(**resp.json())
        return FinetuningService(client=self._client, entity=self.entity, name=obj.name)
