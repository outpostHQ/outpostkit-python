from typing import Any, Dict, List, Optional, TypedDict, Union

from fastapi.requests import Request
from generic.tasks.common import AbstractInferenceHandler
from pydantic import BaseModel
from transformers import SquadExample


class SquadExampleJSONInput(BaseModel):
    qas_id: Any
    question_text: Any
    context_text: Any
    answer_text: Any
    start_position_character: Any
    title: Any
    answers: Any
    is_impossible: Any


class QuestionAnsweringJSONInput(BaseModel):
    args: Union[SquadExampleJSONInput, List[SquadExampleJSONInput]]
    X: Union[SquadExampleJSONInput, List[SquadExampleJSONInput]]
    data: Union[SquadExampleJSONInput, List[SquadExampleJSONInput]]
    question: Union[str, List[str]]
    context: Union[str, List[str]]
    topk: Optional[int] = None
    doc_stride: Optional[int] = None
    max_answer_len: Optional[int] = None
    max_seq_len: Optional[int] = None
    max_question_len: Optional[int] = None
    handle_impossible_answer: Optional[bool] = None
    align_to_words: Optional[bool] = None
    kwargs: Optional[Dict[str, Any]] = {}


class QuestionAnsweringInferenceInput(TypedDict):
    args: Union[SquadExample, List[SquadExample]]
    X: Union[SquadExample, List[SquadExample]]
    data: Union[SquadExample, List[SquadExample]]
    kwargs: Dict[str, Any]


class QuestionAnsweringInference(BaseModel):
    score: float
    start: int
    end: int
    answer: str


class QuestionAnsweringInferenceHandler(AbstractInferenceHandler):
    def convert_to_squad(self, n: Dict[str, Any]) -> SquadExample:
        return SquadExample(**n)

    async def parse_request(
        self, request: Request, temp_dir: str
    ) -> QuestionAnsweringInferenceInput:
        data = await request.json()
        body = QuestionAnsweringJSONInput.model_validate(**data)

        if isinstance(body.args, List):
            body.args = map(self.convert_to_squad, body.args)
        else:
            body.args = self.convert_to_squad(body.args)

        if isinstance(body.X, List):
            body.X = map(self.convert_to_squad, body.X)
        else:
            body.X = self.convert_to_squad(body.X)

        if isinstance(body.data, List):
            body.data = map(self.convert_to_squad, body.data)
        else:
            body.data = self.convert_to_squad(body.data)

        return QuestionAnsweringInferenceInput(
            args=body.args,
            X=body.X,
            data=body.data,
            kwargs=dict(
                question=body.question,
                context=body.context,
                topk=body.topk,
                doc_stride=body.doc_stride,
                max_answer_len=body.max_answer_len,
                max_seq_len=body.max_seq_len,
                max_question_len=body.max_question_len,
                handle_impossible_answer=body.handle_impossible_answer,
                align_to_words=body.align_to_words,
                **body.kwargs,
            ),
        )

    # async def infer(
    #     self, data: QuestionAnsweringInferenceInput
    # ) -> List[QuestionAnsweringInference]:
    #     return self.pipeline(**data)

    # async def respond(self, output: List[QuestionAnsweringInference]):
    #     json_compatible_output = jsonable_encoder(output)
    #     return JSONResponse(content=json_compatible_output)
