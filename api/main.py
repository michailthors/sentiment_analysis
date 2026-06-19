from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = FastAPI()

# Load model
model = DistilBertForSequenceClassification.from_pretrained('michaelthors/sentiment-analysis-distilbert')
tokenizer = DistilBertTokenizer.from_pretrained('michaelthors/sentiment-analysis-distilbert')
model.eval()

class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API"}

@app.post("/predict")
def predict(input: TextInput):
    inputs = tokenizer(
        input.text,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = probs.argmax().item()
        confidence = probs.max().item()
    
    label = "positive" if pred == 1 else "negative"
    
    return {
        "text": input.text,
        "sentiment": label,
        "confidence": round(confidence, 4)
    }