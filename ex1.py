import argparse
import torch
import numpy as np
import evaluate
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# Global metric for MRPC
metric = evaluate.load("glue", "mrpc")


def compute_metrics(eval_preds):
    """Helper function to calculate accuracy during training/evaluation."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def load_and_prepare_data(tokenizer, args):
    """Loads MRPC, tokenizes it, and applies sample limits."""
    dataset = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # Truncate to max length allowed by bert-base-uncased
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

    # Apply tokenization and rename 'label' to 'labels' for Trainer compatibility
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Slice datasets based on CLI arguments
    train_ds = tokenized_datasets["train"]
    if args.max_train_samples != -1:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))

    eval_ds = tokenized_datasets["validation"]
    if args.max_eval_samples != -1:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    test_ds = tokenized_datasets["test"]
    if args.max_predict_samples != -1:
        test_ds = test_ds.select(range(min(args.max_predict_samples, len(test_ds))))

    return train_ds, eval_ds, test_ds


def run_training(args, train_ds, eval_ds, tokenizer, data_collator):
    """Initializes model and WandB, runs the training loop, and logs results."""
    wandb.init(project="anlp_ex1", config=vars(args))

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.model_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",  # שינוי כאן
        save_strategy="no",
        logging_steps=1,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Calculating Training Accuracy...")
    train_results = trainer.evaluate(eval_dataset=train_ds)
    print(f"Final Train Metrics: {train_results}")

    # Log validation accuracy to res.txt
    eval_results = trainer.evaluate()
    eval_acc = eval_results.get("eval_accuracy", 0.0)

    with open("res.txt", "a") as f:
        f.write(
            f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {eval_acc:.4f}\n")

    # Save model for future do_predict calls
    trainer.save_model(args.model_path)
    wandb.finish()


def run_prediction(args, test_ds, data_collator):
    """Loads a fine-tuned model and generates predictions.txt for the test set."""
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.eval()
    # Using Trainer.predict is more efficient (supports batching) than a for-loop
    predict_args = TrainingArguments(output_dir="./temp", per_device_eval_batch_size=args.batch_size, report_to="none")
    trainer = Trainer(model=model, args=predict_args, data_collator=data_collator)

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)

    # Output format: sentence1###sentence2###label
    with open("predictions.txt", "w", encoding="utf-8") as f:
        for i, label in enumerate(preds):
            s1 = test_ds[i]["sentence1"]
            s2 = test_ds[i]["sentence2"]
            f.write(f"{s1}###{s2}###{label}\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="./results")

    args = parser.parse_args()

    # Pre-load shared components
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Get the prepared datasets
    train_ds, eval_ds, test_ds = load_and_prepare_data(tokenizer, args)

    if args.do_train:
        run_training(args, train_ds, eval_ds, tokenizer, data_collator)

    if args.do_predict:
        run_prediction(args, test_ds, data_collator)


if __name__ == "__main__":
    main()