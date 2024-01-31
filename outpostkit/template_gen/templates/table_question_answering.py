import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypedDict, Union

import pandas as pd
import requests
from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from generic.tasks.common import AbstractInferenceHandler
from generic.tasks.utils import (
    form_field_to_boolean,
    form_field_to_json,
)
from pydantic import BaseModel
from starlette.datastructures import UploadFile


class TableQuestionAnsweringJSONInput(BaseModel):
    table: Union[str, Dict]
    query: Union[str, List[str]]
    sequential: Optional[bool] = None
    padding: Optional[Union[bool, str]] = None
    truncation: Optional[Union[bool, str]] = None
    kwargs: Optional[Dict[str, Any]] = {}


class TableQuestionAnsweringInferenceInput(TypedDict):
    table: Type[pd.DataFrame]
    query: Union[str, List[str]]
    sequential: Optional[bool]
    padding: Optional[Union[bool, str]]
    truncation: Optional[Union[bool, str]]
    kwargs: Dict[str, Any]


class TableQuestionAnsweringInference(TypedDict):
    answer: str
    coordinates: List[Tuple[int, int]]
    cells: List[str]
    aggregator: str


class TableQuestionAnsweringInferenceHandler(AbstractInferenceHandler):
    def download_and_load_dataframe(url: str):
        try:
            # Download the file from the URL
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful

            # Determine file format based on the URL or response content type
            file_extension = url.split(".")[
                -1
            ].lower()  # Extract file extension from URL
            content_type = response.headers.get("content-type", "").lower()

            if "text/csv" in content_type or file_extension == "csv":
                df = pd.read_csv(io.StringIO(response.text))
            elif "application/json" in content_type or file_extension == "json":
                df = pd.read_json(io.StringIO(response.text))
            elif "application/vnd.ms-excel" in content_type or file_extension in [
                "xls",
                "xlsx",
            ]:
                df = pd.read_excel(io.BytesIO(response.content))
            elif file_extension == "h5":
                df = pd.read_hdf(io.BytesIO(response.content))
            elif file_extension == "parquet":
                df = pd.read_parquet(io.BytesIO(response.content))
            else:
                raise ValueError(f"Unsupported file format: {file_extension}.")
            return df
        except requests.exceptions.HTTPError as errh:
            raise ValueError(
                f"HTTP Error while downloading table file: {errh}"
            ) from errh
        except requests.exceptions.ConnectionError as errc:
            raise ValueError(
                f"Error Connecting while downloading table file: {errc}"
            ) from errc
        except requests.exceptions.Timeout as errt:
            raise ValueError(
                f"Timeout Error while downloading table file: {errt}"
            ) from errt
        except requests.exceptions.RequestException as err:
            raise ValueError(
                f"Request Error while downloading table file: {err}"
            ) from err

    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> TableQuestionAnsweringInferenceInput:
        """
        ContentType: multipart/form-data, application/json
        Args:
            - table : pandas DataFrame or dictionary that will be converted to a DataFrame containing all the table values.
            - query : Query or list of queries that will be sent to the model alongside the table.
            - sequential : Whether to do inference sequentially or as a batch.
            - padding : Activates and controls padding.
            - truncation : Activates and controls truncation.
            - kwargs : Additional arguments.
        Returns: {table:Type[pd.DataFrame], query:Union[str,List[str]], sequential:Optional[bool], padding:Optional[Union[bool,str]], truncation:Optional[Union[bool,str]], kwargs:Dict[str,Any]}
        """
        if request.headers["content-type"].startswith("multipart/form-data"):
            form = await request.form()
            raw_table = form.get("table")
            queries = form.getlist("query")

            if all(isinstance(q, str) for q in queries) is not True:
                raise HTTPException(status_code=400, detail="Unable to parse query")

            table: Type[pd.DataFrame]
            if isinstance(raw_table, UploadFile):
                extention = Path(raw_table.filename).suffix
                try:
                    if extention == ".csv":
                        table = pd.read_csv(io.BytesIO(await raw_table.read()))
                    elif extention == ".json":
                        table = pd.read_json(io.BytesIO(await raw_table.read()))
                    elif extention == ".hdf5":
                        table = pd.read_hdf(io.BytesIO(await raw_table.read()))
                    elif extention == ".parquet":
                        table = pd.read_parquet(io.BytesIO(await raw_table.read()))
                    elif (
                        extention == ".xls"
                        or extention == ".xlsx"
                        or extention == ".odf"
                        or extention == ".ods"
                    ):
                        table = pd.read_excel(io.BytesIO(await raw_table.read()))
                    else:
                        raise HTTPException(
                            status_code=400, detail="Invalid file format"
                        )
                except Exception as e:
                    raise HTTPException(status_code=400, detail=str(e)) from e
            elif isinstance(table, str):
                table = self.download_and_load_dataframe(table)

            else:
                raise HTTPException(status_code=400, detail="No table provided.")

            return TableQuestionAnsweringInferenceInput(
                table=table,
                query=queries,
                kwargs=dict(
                    sequential=form_field_to_boolean(
                        "sequential", form.get("sequential")
                    ),
                    padding=form_field_to_boolean("padding", form.get("padding")),
                    truncation=form_field_to_boolean(
                        "truncation", form.get("truncation")
                    ),
                    **form_field_to_json(
                        name="kwargs", value=form.get("kwargs"), on_none={}
                    ),
                ),
            )

        elif request.headers["content-type"].startswith("application/json"):
            data = await request.json()
            body = TableQuestionAnsweringJSONInput.model_validate(data)
            table: Union[Dict, pd.DataFrame] = body.table
            if isinstance(table, str):
                table = self.download_and_load_dataframe(table)
            return TableQuestionAnsweringInferenceInput(
                table=body.table,
                query=body.query,
                kwargs=dict(
                    sequential=body.sequential,
                    padding=body.padding,
                    truncation=body.truncation,
                    **body.kwargs,
                ),
            )

        else:
            raise HTTPException(status_code=400, detail="Invalid content type")

    async def infer(self, data: TableQuestionAnsweringInferenceInput) -> Union[
        List[TableQuestionAnsweringInference],
        List[List[TableQuestionAnsweringInference]],
    ]:
        return self.pipeline(**data)

    async def respond(
        self,
        output: Union[
            List[TableQuestionAnsweringInference],
            List[List[TableQuestionAnsweringInference]],
        ],
    ):
        json_compatible_output = jsonable_encoder(output)
        return JSONResponse(content=json_compatible_output)
