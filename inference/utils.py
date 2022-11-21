from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def download_artifacts():
    '''
        Downloads the huggingface model for deployment. 

        We can choose to download it once and resue it everytime to save time.
        But a model update will not download the latest version of that model for deployment.
    '''
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForSeq2SeqLM.from_pretrained("distilbert-base-cased-distilled-squad")

    return tokenizer, model
