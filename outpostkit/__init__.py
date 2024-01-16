import os

from outpostkit.client import Client
from outpostkit.pagination import async_paginate as _async_paginate
from outpostkit.pagination import paginate as _paginate

default_client = Client(
    api_token=os.environ.get("OUTPOST_API_TOKEN")
)

run = default_client.run
async_run = default_client.async_run

stream = default_client.stream
async_stream = default_client.async_stream

paginate = _paginate
async_paginate = _async_paginate

collections = default_client.collections
hardware = default_client.hardware
deployments = default_client.deployments
models = default_client.models
predictions = default_client.predictions
trainings = default_client.trainings
inferences =  default_client.inferences
