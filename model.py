import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, Trainer, TrainingArguments

class Model:
    def __init__(self, model_name = "bert-base-uncased"):
        """
        Initialize and return the pre-trained QA model.
        Args:
        - model_name (str): The name of the pre-trained model to use.
         Define and return the training arguments for the Trainer.

        Args:
        - output_dir (str): Directory to save training outputs.
        - batch_size (int): Batch size for training and evaluation.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for the optimizer.
        """

        # Load a pre-trained QA model
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define training arguments (Team 4 can edit these hyperparameters)
        self.training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01
        )
    
    # Function for reading the data from csv
    def data_reader(self):
        data = pd.read_csv("processed_data.csv")
        data = data.dropna()

        questions = [q.strip() for q in data["processed_question"]]

        inputs = self.tokenizer(
            questions, data["processed_answer_text"].tolist(), max_length=384, truncation=True, padding="max_length", return_tensors="pt"
        )

        return inputs

    def train_model(self, tokenized_train, tokenized_valid):
    # Initialize Trainer
        trainer = Trainer(
            model = self.model,
            args = self.training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid
        )

        # Train the model
        trainer.train()
        return trainer # this returned object has to be used for predicting on validation or test data.

"""
Notes for the next team: 
- call the data_reader function to get tokenised data for train and validation
- next u can call the train_model function for making the training happen.
- do split the data into train and validation.
- for the model training team (team 4), finding the optimal value for hyper parameters is the main goal
"""