import gradio as gr
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load model + tokenizer (expects distilbert_sentiment/ folder in same repo)
model_path = "./distilbert_sentiment"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None)

def predict_sentiment(text):
    preds = pipe(text)[0]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Positive"}
    return {label_map[p["label"]]: float(p["score"]) for p in preds}

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type a review here..."),
    outputs="label",
    title="Sentiment Classifier (DistilBERT)",
    description="Enter text to classify sentiment as Positive or Negative"
)

if __name__ == "__main__":
    demo.launch()
