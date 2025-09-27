# Sentiment Classifier with DistilBERT

This project builds a **binary sentiment classifier** using both classical ML baselines and modern Transformer models.  
While trained on the Amazon Reviews dataset, the pipeline is fully generalizable to other review or text classification tasks.  
The final model is deployed on [Hugging Face Spaces](https://karthik89v-sentiment-classifier-nlp.hf.space).

---

## 🔑 Key Features
- End-to-end NLP pipeline:
  - Data preprocessing (cleaning, tokenization, lemmatization).
  - Feature engineering with **TF-IDF**.
  - Baseline models with **Naive Bayes**.
  - Fine-tuned **DistilBERT Transformer**.
- High performance:
  - TF-IDF + Naive Bayes → ~88.6% accuracy.
  - DistilBERT fine-tuned → ~91.3% accuracy.
- Error analysis of false positives/negatives.
- Deployed with Gradio on Hugging Face Spaces.

Karthik Velavarthypathi
