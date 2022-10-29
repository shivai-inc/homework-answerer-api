from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

class Questions(BaseModel):
    question_text: str
    context: str
    answer: str

question_answerer_api = FastAPI()

@question_answerer_api.post("/fetch_answers/")
async def create_answer(question: Questions):
    question_dict = question.dict()
    result = question_answerer(question = question.question_text, context=question.context)
    question_dict.update({"answer":str(result['answer'])})
    return question_dict


@question_answerer_api.get("/")
async def root():
    return {"message": "Welcome to Homework Question Solver API"}
