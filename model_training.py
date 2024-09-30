import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from datasets import Dataset

class Model:
    def __init__(self, model_name="bert-base-uncased", epochs=3):
        """
        Initialize the pre-trained BERT-based QA model and define training arguments.
        
        Args:
        - model_name (str): Name of the pre-trained model to use.
        - epochs (int): Number of epochs for training.
        """
        # Load the pre-trained model and tokenizer
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            weight_decay=0.01
        )

    def data_reader(self):
        """
        Read and preprocess the data from processed_data.csv.
        
        Returns:
        - inputs (dict): Tokenized inputs including input_ids and attention_masks.
        """
        # Read the CSV data file
        data = pd.read_csv("processed_data.csv")
        data = data.dropna()  # Drop any missing data
        
        # Get questions and answers from the dataset
        questions = [q.strip() for q in data["processed_question"]]
        answers = data["processed_answer_text"].tolist()

        # Tokenize the questions and answers
        inputs = self.tokenizer(
            questions, answers, max_length=384, truncation=True, padding="max_length", return_tensors="pt"
        )
        
        return inputs

    def prepare_data(self):
        """
        Prepare the data by splitting it into training and validation sets.
        
        Returns:
        - train_dataset (Dataset): Training dataset.
        - val_dataset (Dataset): Validation dataset.
        """
        # Read and tokenize data
        inputs = self.data_reader()
        
        # Split input data into train and validation sets (80% train, 20% validation)
        input_ids_train, input_ids_val, attention_mask_train, attention_mask_val = train_test_split(
            inputs['input_ids'], inputs['attention_mask'], test_size=0.2, random_state=42
        )

        # Create torch Dataset objects for training and validation
        train_dataset = Dataset.from_dict({
            'input_ids': input_ids_train, 
            'attention_mask': attention_mask_train
        })
        val_dataset = Dataset.from_dict({
            'input_ids': input_ids_val, 
            'attention_mask': attention_mask_val
        })
        
        return train_dataset, val_dataset

    def train_model(self):
        """
        Train the model using the Trainer class, evaluate it on the validation set,
        and return the trainer object and evaluation metrics.
        
        Returns:
        - trainer (Trainer): Trainer object used for training the model.
        - eval_metrics (dict): Evaluation metrics from the validation set.
        """
        # Prepare training and validation datasets
        train_dataset, val_dataset = self.prepare_data()

        # Initialize Trainer for the model
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train the model
        trainer.train()

        # Evaluate the model on the validation set
        eval_metrics = trainer.evaluate()
        print(f"Evaluation metrics: {eval_metrics}")
        
        return trainer, eval_metrics

# Instantiate the model class and train it with 3 epochs
qa_model = Model(epochs=3)
trainer, eval_metrics = qa_model.train_model()
