import streamlit as st
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np

# Load the model and tokenizer
with open("trained_model.pkl", "rb") as f:
    model = pickle.load('BtechAI_B3_Batch/trained_model.pkl')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def answer_question(question, context):
    inputs = tokenizer(
        question, context, max_length=384, truncation=True, padding="max_length", return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits

    start_pos = torch.argmax(start_logits)
    end_pos = torch.argmax(end_logits) + 1

    answer_tokens = inputs["input_ids"][0][start_pos:end_pos]
    answer = tokenizer.decode(answer_tokens)

    return answer

# Streamlit interface
st.title("Question Answering System")

# Chatbox to input the question
user_question = st.text_input("Ask a question:")

# Static context (or this can be dynamically added/changed)
context = """
    COntext
"""

if user_question:
    # Get the model's answer
    answer = answer_question(user_question, context)
    
    # Display the model's answer below the text box
    st.write(f"**Answer:** {answer}")
