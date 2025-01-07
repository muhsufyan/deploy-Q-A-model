from click import option
from fastapi import FastAPI
from fastapi.params import Body
from pydantic import BaseModel

# load model
from transformers import pipeline, AutoTokenizer
from app import model
# Example of inference
model = model.CustomBertForQuestionAnswering.from_pretrained("practice-ac/training_question_answer")
tokenizer = AutoTokenizer.from_pretrained("practice-ac/tokenizer-model-for-deploy-Q-A-model")
nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)


app = FastAPI(title="Dokumentasi untuk api")

class Data(BaseModel):
    Question: str
    Context: str


@app.post("/predict", tags=["inference model"], summary=["predict"])
async def prediction(capData: Data):
    QA_input = {
    'question': capData.Question,
    'context': capData.Context
    }
    predict = nlp(QA_input)
    return {
        "answer": predict['answer'],
        "data": [predict]
  }
