from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import torch
import numpy as np
from datasets import Dataset

class Model:
    def __init__(self, model_name="bert-base-uncased", epochs=3):
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Reduce batch size to prevent out-of-memory errors
        self.training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=8,  # Reduced batch size
            per_device_eval_batch_size=8,   # Reduced eval batch size
            num_train_epochs=epochs,
            weight_decay=0.01,
            gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
            no_cuda=False                   # Force CPU if necessary (set to True if GPU memory issue persists)
        )

    def data_reader(self):
        data = pd.read_csv("BtechAI_B3_Batch/processed_data.csv")
        data = data.dropna()
        questions = [q.strip() for q in data["processed_question"]]
        answers = data["processed_answer_text"].tolist()

        inputs = self.tokenizer(
            questions, answers, max_length=384, truncation=True, padding="max_length", return_tensors="pt"
        )

        start_positions = []
        end_positions = []
        for i, answer in enumerate(answers):
            answer_tokens = self.tokenizer.tokenize(answer)
            answer_ids = self.tokenizer.convert_tokens_to_ids(answer_tokens)
            try:
                start_pos = inputs["input_ids"][i].tolist().index(answer_ids[0])
                end_pos = start_pos + len(answer_ids) - 1
            except ValueError:
                start_pos = -1
                end_pos = -1
            start_positions.append(start_pos)
            end_positions.append(end_pos)
        
        inputs["start_positions"] = torch.tensor(start_positions)
        inputs["end_positions"] = torch.tensor(end_positions)

        return inputs

    def prepare_data(self):
        inputs = self.data_reader()
        train_input_ids, val_input_ids = train_test_split(inputs["input_ids"], test_size=0.2, random_state=42)
        train_attention_mask, val_attention_mask = train_test_split(inputs["attention_mask"], test_size=0.2, random_state=42)
        train_start_positions, val_start_positions = train_test_split(inputs["start_positions"], test_size=0.2, random_state=42)
        train_end_positions, val_end_positions = train_test_split(inputs["end_positions"], test_size=0.2, random_state=42)

        train_dataset = Dataset.from_dict({
            'input_ids': train_input_ids,
            'attention_mask': train_attention_mask,
            'start_positions': train_start_positions,
            'end_positions': train_end_positions
        })
        
        val_dataset = Dataset.from_dict({
            'input_ids': val_input_ids,
            'attention_mask': val_attention_mask,
            'start_positions': val_start_positions,
            'end_positions': val_end_positions
        })
        
        return train_dataset, val_dataset

    def train_model(self):
        train_dataset, val_dataset = self.prepare_data()

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        eval_metrics = trainer.evaluate()
        print(f"Evaluation metrics: {eval_metrics}")
        
        return trainer, eval_metrics
    




    def evaluate_model(self, val_dataset):
        predictions, labels = self.predict(val_dataset)
        start_preds = np.argmax(predictions['start_logits'], axis=-1)
        end_preds = np.argmax(predictions['end_logits'], axis=-1)
        
        start_labels = val_dataset["start_positions"]
        end_labels = val_dataset["end_positions"]

        em = self.exact_match(start_preds, end_preds, start_labels, end_labels)
        f1 = self.compute_f1(start_preds, end_preds, start_labels, end_labels)
        
        print(f"Exact Match (EM): {em}")
        print(f"F1 Score: {f1}")
        
        return {"Exact Match": em, "F1 Score": f1}
    

    # predict function
    def predict(self, dataset):
        trainer = Trainer(model=self.model, args=self.training_args)
        predictions = trainer.predict(dataset)
        return predictions.predictions, dataset

    # exact_match function
    def exact_match(self, start_preds, end_preds, start_labels, end_labels):
        """Calculate exact match (EM) score."""
        em_score = (start_preds == start_labels) & (end_preds == end_labels)
        return np.mean(em_score)
    
    
    def compute_f1(self, start_preds, end_preds, start_labels, end_labels):
        """Calculate F1 score between predicted and true positions."""
        f1_scores = []
        for i in range(len(start_labels)):
            pred_range = set(range(start_preds[i], end_preds[i] + 1))
            true_range = set(range(start_labels[i], end_labels[i] + 1))
            
            common_tokens = pred_range.intersection(true_range)
            if len(common_tokens) == 0:
                f1_scores.append(0)
            else:
                precision = len(common_tokens) / len(pred_range)
                recall = len(common_tokens) / len(true_range)
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
        return np.mean(f1_scores)
    






# Example usage:
qa_model = Model(epochs=3)
trainer, eval_metrics = qa_model.train_model()  # Train the model
train_dataset, val_dataset = qa_model.prepare_data()  # Prepare data
eval_metrics = qa_model.evaluate_model(val_dataset)  # Evaluate the model
