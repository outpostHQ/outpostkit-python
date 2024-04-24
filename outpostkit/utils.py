from datetime import datetime
from typing import Any, Dict, List, Tuple

from outpostkit._types.endpoint import EndpointLogData
from outpostkit._types.finetuning import FinetuningJobLogData


def convert_outpost_date_str_to_date(date_string: str) -> datetime:
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")


def separate_keys(
    dictionary: Dict[str, Any], known_keys: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    known_dict = {}
    unknown_dict = {}

    for key, value in dictionary.items():
        if key in known_keys:
            known_dict[key] = value
        else:
            unknown_dict[key] = value

    return (known_dict, unknown_dict)


def parse_endpoint_log_data(log_data: Dict[str, Any]) -> EndpointLogData:
    known_keys = [
        "level_num",
        "log_type",
        "level",
        "logger_name",
        "message",
        "exc_info",
        "stack_info",
    ]
    (known_dict, extra) = separate_keys(log_data, known_keys=known_keys)
    replica = None
    kube_data = extra.get("kubernetes")
    if kube_data and isinstance(kube_data, dict):
        if "pod_name" in kube_data and isinstance(kube_data["pod_name"], str):
            parts = kube_data["pod_name"].split("-")
            replica = parts[-1]
    return EndpointLogData(**known_dict, replica=replica, extra=extra)


def parse_finetuning_job_log_data(log_data: Dict[str, Any]) -> FinetuningJobLogData:
    known_keys = [
        "level_num",
        "log_type",
        "level",
        "logger_name",
        "message",
        "exc_info",
        "stack_info",
    ]
    (known_dict, extra) = separate_keys(log_data, known_keys=known_keys)
    return FinetuningJobLogData(**known_dict, extra=extra)
