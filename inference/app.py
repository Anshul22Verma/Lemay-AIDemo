from datetime import datetime as dt
from fastapi import Request,FastAPI
from pydantic import BaseModel
import uvicorn

from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from utils import download_artifacts

app = FastAPI()

# use GPU if available
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)


tokenizer, model = download_artifacts
model = model.to(torch_device)


class SummaryRequest(BaseModel):
    context: str
    question: str


def get_answer_for_context(request: dict, tokenizer, model):
    start_time = dt.now()
    inputs = tokenizer(request['question'], request['context'], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return {'question': request['question'], 'summary': outputs, 'prediction_time': f'{(dt.now() - start_time).seconds} seconds'}


@app.get('/')
async def home():
    return {"message": "Lemay-AI Demo"}

@app.post("/answer")
async def getsummary(user_request_in: SummaryRequest):
    payload = {"context": user_request_in.context, "question": user_request_in.question}
    response = get_answer_for_context(payload,tokenizer,model)
    # to ensure the device using which the prediction was made
    response["device"]= torch_device
    return response
