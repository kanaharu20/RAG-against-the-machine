from pydantic import BaseModel
from pydantic.type_adapter import TypeAdapter
from main import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

    @staticmethod
    def load_questions(path: str) -> list[AnsweredQuestion | UnansweredQuestion]:
        with open(path) as fd:
            return TypeAdapter(
                list[AnsweredQuestion | UnansweredQuestion]
                ).validate_json(fd.read())

