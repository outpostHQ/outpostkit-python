import os

from outpostkit.client import Client

default_client = Client(api_token=os.environ.get("OUTPOST_API_TOKEN"))

run = default_client.run
async_run = default_client.async_run

stream = default_client.stream
async_stream = default_client.async_stream
inferences = default_client.inferences
