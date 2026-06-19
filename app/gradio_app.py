import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load model
model = DistilBertForSequenceClassification.from_pretrained('michaelthors/sentiment-analysis-distilbert')
tokenizer = DistilBertTokenizer.from_pretrained('michaelthors/sentiment-analysis-distilbert')
model.eval()

def predict_sentiment(text):
    inputs = tokenizer(
        text,
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
    
    label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    return f"{label} (Confidence: {confidence:.2%})"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, placeholder="Write a movie review here..."),
    outputs=gr.Textbox(label="Sentiment"),
    title="🎬 Movie Sentiment Analysis",
    description="Enter a movie review and the model will predict if it's positive or negative."
)

if __name__ == "__main__":
    demo.launch()