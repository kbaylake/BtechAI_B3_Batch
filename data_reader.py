import pandas as pd

# Function for reading the data from csv
def data_reader(tokenizer):
    data = pd.read_csv("processed_data.csv")
    data = data.dropna()

    questions = [q.strip() for q in data["processed_question"]]

    inputs = tokenizer(
        questions, data["processed_answer_text"].tolist(), max_length=384, truncation=True, padding="max_length", return_tensors="pt"
    )

    return inputs
    