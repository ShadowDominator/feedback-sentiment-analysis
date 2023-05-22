import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

tokenizer_review_feedback_sentiment = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model_review_feedback_sentiment = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


def review_feedback_sentiment(text, tokenizer, model):
    inputs = tokenizer.encode_plus(text, padding='max_length', max_length=512, return_tensors="pt")
    with torch.no_grad():
        result = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = result.logits.detach()
        probs = torch.softmax(logits, dim=1).detach().numpy()[0]
        categories = ['Terrible', 'Poor', 'Average', 'Good', 'Excellent']
        output_dict = {}
        for i in range(len(categories)):
            output_dict[categories[i]] = [round(float(probs[i]), 2)]
    return output_dict

def review_feed_back(text):
    result = review_feedback_sentiment(text,tokenizer_review_feedback_sentiment,model_review_feedback_sentiment)
    return result

with gr.Blocks(title="Feedback",css="footer {visibility: hidden}") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Review Feedback sentiment")
            with gr.Row():           
                with gr.Column():
                    inputs = gr.TextArea(label="sentence",value="I'm so impressed with your product! It's exactly what I needed and it's working great.",interactive=True)
                    btn = gr.Button(value="RUN")
                with gr.Column():
                    output = gr.Label(label="output")
                btn.click(fn=review_feed_back,inputs=[inputs],outputs=[output])
demo.launch()