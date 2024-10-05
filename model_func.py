from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments

def model(model_name = "bert-base-uncased"):
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
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define training arguments (Team 4 can edit these hyperparameters)
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=3e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01
        )

        return model, tokenizer, training_args
    