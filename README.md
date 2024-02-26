# Outpost Kit Python client

This is a Python client for [Outpost](https://outpost.run). It lets you run models from your Python code or Outpost Notebook, and do various other things on Outpost.

## Requirements

- Python 3.8+

## Install

```sh
pip install outpostkit
```

## Authenticate

Before running any Python scripts that use the API, you need to set your Outpost API token in your environment.

Get your token from [outpost.run/](https://outpost.run/) and set it as an environment variable:

```
export OUTPOST_API_TOKEN=<your token>
```

## Check the version
```py
import outpostkit
print(outpostkit.__version__)
```
`0.0.40`


## Create a client

```py
from outpostkit import Client
import os
client = Client(api_token=os.environ.get('OUTPOST_API_TOKEN'))
```

## Get details of the authenticated user
```py
print(client.user)
```
```
UserDetails(id='lqiuxjj6okbzt72j1eyk5vn2', createdAt='2023-12-08T13:26:01.277Z', updatedAt='2024-01-08T12:17:45.927Z', stats=UserStats(followers_count=1, following_count=2), avatarUrl='https://avatars.outpost.run/e/lqiuxjj6okbzt72j1eyk5vn2', name='aj-ya', bio=None, socials=None, displayName='Ajeya Bhat')
```

## Create an Endpoint 


The easiest way to create an endpoint is by providing just the model and let us handle the server specifics.
We use the model's task type, library and other details to generate the template that serves it.

```py
from outpostkit import Endpoints
from outpostkit._types.endpoints import EndpointAutogeneratedTemplateConfig, EndpointAutogeneratedHFModelDetails

client = Client(api_token=os.environ.get('OUTPOST_API_TOKEN'))
endpoints_client = Endpoints(client=client, entity=client.user.name)

template = EndpointAutogeneratedTemplateConfig(modelSource="huggingface",huggingfaceModel=EndpointAutogeneratedHFModelDetails(id="Falconsai/text_summarization"))  
endpoint = endpoints_client.create(templateConfig=template)
```
To import gated/private models from Hugging Face, you can add your Hugging Face key as a third party token and provide its `id` in the config.
```py
EndpointAutogeneratedHFModelDetails(id="Falconsai/text_summarization",keyId="<thirdPartyTokenId>")
```

If you have a specific revision of the model that you want to deploy, provide the revision in the `revision` field.

```py
template = EndpointAutogeneratedTemplateConfig(modelSource="outpost", revision="df5ef1a0e2d2579726d74b5d617b17c7049c5a89",outpostModel=EndpointAutogeneratedOutpostModelDetails(id="aj-ya/text-gen")) 
```
The horizontal scaling configurations of the endpoint are based on the number of requests to the prediction request path.
You can tweak the settings at creation too.
```py
from outpostkit._types.endpoint import ReplicaScalingConfig
scaling_config = ReplicaScalingConfig(min=0,max=1,scaledownPeriod=900,targetPendingRequests=100) # Defaults
endpoint = endpoints_client.create(templateConfig=template,replica_scaling_config = scaling_config)
```

`scaledownPeriod`: The period to wait after the last reported active before scaling the resource back to 0.
`targetPendingRequests`: This is the number of pending (or in-progress) requests that your application needs to have before it is scaled up. Conversely, if your application has below this number of pending requests, it will scaled down.

## Deploy the endpoint
once you create the endpoint, you need to deploy it.
```py
endpoint.deploy()
```
To reflect the updates in configurations of the endpoint, it must be redeployed.

Once it is deployed... a dummy request is sent to the prediction path to trigger the scale up. Thus it is available for atleast for the duration of the `scaledownPeriod`. To not wake up the deployment. you can use set it the deployment parameters.
```py
endpoint.deploy(wakeup=False)
```

## Get prediction from the endpoint
Once the endpoint is available, you can test the predictions over HTTP.
> The requests to the prediction path must be authenticated with outpost access token.

You can also use our prediction client for that. (which uses our main client for the token.)
```py
pred_client = endpoint.get_prediction_client()
resp = pred_client.infer(json={
    "documents":"""Imagine you are standing in the middle of a room with no windows, doors or lights. What do you see? Well, nothing because there’s no light. Now imagine you pull out a flashlight and turn it on. The light from the flashlight moves in a straight line. When that beam of light hits an object, the light bounces off that item and into your eyes, allowing you to see whatever is inside the room.
    All light behaves just like that flashlight — it travels in a straight line. But, light also bounces off of objects, which is what allows us to see and photograph objects. When light bounces off an object, it continues to travel in a straight line, but it bounces back at the same angle that it comes in at. That means light rays are essentially bouncing everywhere in all kinds of different directions. The first camera was essentially a room with a small hole on one side wall. Light would pass through that hole, and since it’s reflected in straight lines, the image would be projected on the opposite wall, upside down. While devices like this existed long before true photography, it wasn’t until someone decided to place material that was sensitive to light at the back of that room that photography was born. When light hit the material, which through the course of photography’s history was made up of things from glass to paper, the chemicals reacted to light, etching an image in the surface."""
})

print(resp.json())
```
```json
[
    {
        "summary_text": "Imagine you are standing in the middle of a room with no windows, doors or lights . When that beam of light hits an object, the light bounces off that item and into your eyes, allowing you to see whatever is inside the room . All light behaves just like that flashlight — it travels in a straight line, but it bounces back at the same angle that it comes in at . That means light rays are essentially bouncing everywhere in all kinds of different directions ."
    }
]
```

## Development
See [CONTRIBUTING.md](CONTRIBUTING.md)
