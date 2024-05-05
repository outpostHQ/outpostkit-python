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
`0.0.61`


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
endpoint = endpoints_client.create(template=template)
```

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


## Endpoint Creation Payload
```
    def create(
        self,
        template: Union[
            EndpointAutogeneratedTemplateConfig, EndpointCustomTemplateConfig
        ],
        container: Optional[EndpointPrebuiltContainerDetails] = None,
        hardware_instance: str = "e2-standard-2",
        task_type: Optional[str] = None,
        name: Optional[str] = None,
        secrets: Optional[List[EndpointSecret]] = None,
        visibility: ServiceVisibility = ServiceVisibility.public,
        replica_scaling_config: Optional[ReplicaScalingConfig] = None,
    ) -> Endpoint: ...
```
An endpoint server needs to know certain things like model loading, prediction request handling, exception handling, etc. and these things vary with each usecase.
Thus the server follows a template which tells it how to behave.

### Template configuration
You can either let us autogenerate templates based on the model information, or create a custom template yourself.

#### Configs to Autogenerate Template
Currently we can autogenerate templates models stored at 'Outpost' or 'Hugging Face'.

Task types supported
--TODO: List all task types supported--

To Import a model stored at Outpost, you can directly use:

```py
from outpostkit._types.endpoints import EndpointAutogeneratedTemplateConfig, EndpointAutogeneratedOutpostModelDetails

template = EndpointAutogeneratedTemplateConfig(modelSource="outpost",outpostModel=EndpointAutogeneratedOutpostModelDetails(id="aj-ya/text-gen"))
```

If you have a specific revision of the model that you want to deploy, provide the revision in the `revision` field.

```py
template = EndpointAutogeneratedTemplateConfig(modelSource="outpost", revision="df5ef1a0e2d2579726d74b5d617b17c7049c5a89",outpostModel=EndpointAutogeneratedOutpostModelDetails(id="aj-ya/text-gen")) 
```

To import gated/private models from Hugging Face, you can add your Hugging Face key as a third party token and provide its `id` in the config.
```py
from outpostkit._types.endpoints import EndpointAutogeneratedTemplateConfig, EndpointAutogeneratedHFModelDetails
hf_model = EndpointAutogeneratedHFModelDetails(id="Falconsai/text_summarization",keyId="<thirdPartyTokenId>")
template = EndpointAutogeneratedTemplateConfig(modelSource="huggingface",huggingfaceModel=hf_model) 
```

#### Create a custom template
A Template class needs to mainly define model initialization and prediction request handling.
For demonstration purposes, lets create a template file for the `openai/shap-e` model.

First of all we need to create a class and the load the model at the initialization phase.
Then, we need to define the request handler for the `/predict` route. this is done by defining
the `predict` member function.
This function itself acts as the handler, thus you can define any parameters that FastAPI supports to the function. Here we will use a pydantic class to validate the request json body and get generation arguements. (ref: https://fastapi.tiangolo.com/tutorial/body/)
Finally, we would like to also stream the output GIF, for this we will use the `StreamingResponse` object by fastapi.

```py
from io import BytesIO
from typing import List
from diffusers import ShapEPipeline
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL.Image import Image


class Item(BaseModel):
    prompt: str
    frame_size: int = 256
    num_inference_steps: int = 64
    guidance_scale: float = 15


def pil_gif_resp(image: List[Image]) -> StreamingResponse:
    temp = BytesIO()
    image[0].save(
        temp,
        save_all=True,
        append_images=image[1:],
        optimize=False,
        duration=100,
        loop=0,
    )
    return StreamingResponse(temp, media_type="image/gif")


class ShapEHandler:
    pipeline: ShapEPipeline

    def __init__(self) -> None:
        ckpt_id = "openai/shap-e"
        self.pipeline = ShapEPipeline.from_pretrained(ckpt_id).to("cuda")

    def predict(self, item: Item):
        images = self.pipeline(
            item.prompt,
            guidance_scale=item.guidance_scale,
            num_inference_steps=item.num_inference_steps,
            frame_size=item.frame_size,
        ).images
        return pil_gif_resp(images[0])
```

##### Installing extra packages and modules
If your application needs a specific python package or system dependency that is not already installed in the container image (--TODO-- docs showing the list of prebuilt container images and the packages installed there.).
Then, you can define these members in the Template class

```py
class Template:
    # extra system dependencies required
    system_dependencies: List[str] = ['curl']

    # extra python packages required
    python_requirements: List[str] = ['gif==23.0']
    ...
```

##### Exception Handling

To define exception handling outside the prediction handler, you can extend the default expection handling done by the server like this:
```py
from fastapi.responses import JSONResponse

async def generic_exception_handler(_, exc: Exception):
    return JSONResponse(
        json.dumps({"error": str(exc), "type": "unhandled_error"}),
        status_code=500,
    )

class Template:
    # define custom exception handlers for the fastapi app
    exception_handlers: Dict[Union[int, Type[Exception]], Callable] = dict({
        Exception: generic_exception_handler
    })
    ...
```


### Container configuration

if youre already using a prebuilt template, most of the times, you wont need to define this. It is already selected based on the library and task type.

But you can manually configure this as well.
Currently, you can only use any one of many prebuilt containers that are provided by outpost.


Namely,
--TODO-- list all container images here.

To use the tensorflow image with pytorch loaded, with some extra configs, use:
```
from outpostkit import EndpointPrebuiltContainerDetails
container = EndpointPrebuiltContainerDetails(name="transformers-pt", configs = {torch_dtype:'float32'})
```


### Scaling Configuration

The horizontal scaling configurations of the endpoint are based on the number of requests to the prediction request path.
You can tweak the settings at creation too.
```py
from outpostkit._types.endpoint import ReplicaScalingConfig
scaling_config = ReplicaScalingConfig(min=0,max=1,scaledownPeriod=900,targetPendingRequests=100) # Defaults
endpoint = endpoints_client.create(template=template,replica_scaling_config = scaling_config)
```

`scaledownPeriod`: The period to wait after the last reported active before scaling the resource back to 0.
`targetPendingRequests`: This is the number of pending (or in-progress) requests that your application needs to have before it is scaled up. Conversely, if your application has below this number of pending requests, it will scaled down.
