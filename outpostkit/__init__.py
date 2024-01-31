import os

from outpostkit.client import Client

default_client = Client(api_token=os.environ.get("OUTPOST_API_TOKEN"))

inferences = default_client.inferences
