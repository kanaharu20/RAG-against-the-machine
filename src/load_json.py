from pydantic import BaseModel
from pydantic.type_adapter import TypeAdapter
from __main__ import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    def load_questions(
        self, path: str
            ) -> None:
        with open(path) as fd:
            self.rag_questions = TypeAdapter(
                list[AnsweredQuestion | UnansweredQuestion]
                ).validate_json(fd.read())
