from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn

from transformers.pipelines import pipeline
from transformers import DistilBertTokenizer, DistilBertModel
import torch

app = FastAPI()

# use GPU if available
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


huggingface_model = "distilbert-base-cased-distilled-squad"
question_answerer = pipeline("question-answering", model=huggingface_model)


class SummaryRequest(BaseModel):
    context: str
    question: str


def get_answer_for_context(request: dict, question_answerer: pipeline):
    result = question_answerer(question=request['question'], context=request['context'])
    return {'question': request['question'], 'answer': result}


@app.get('/')
async def home():
    return {"message": "Lemay-AI Demo"}

@app.post("/answer")
async def getsummary(user_request_in: SummaryRequest):
    payload = {"context": user_request_in.context, "question": user_request_in.question}
    response = get_answer_for_context(payload, question_answerer)
    # to ensure the device using which the prediction was made
    response["device"]= torch_device
    return response
