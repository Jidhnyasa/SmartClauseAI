from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
import torch

# Load datasets directly from Hugging Face
case_hold = load_dataset("coastalcph/lex_glue", "case_hold")['train']
ecthr_a = load_dataset("coastalcph/lex_glue", "ecthr_a")['train']
ecthr_b = load_dataset("coastalcph/lex_glue", "ecthr_b")['train']

# Combine all datasets for training
combined_dataset = concatenate_datasets([case_hold, ecthr_a, ecthr_b])

# Filter out bad data
def is_valid(example):
    return example["endings"] is not None and all(e is not None for e in example["endings"])

filtered_dataset = combined_dataset.filter(is_valid)

# Print columns to verify
print(filtered_dataset.column_names)

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

# Tokenization function
def tokenize_function(batch):
    contexts = batch["context"]
    endings = batch["endings"]
    labels = batch["label"]

    first_sentences = []
    second_sentences = []

    for context, option_list in zip(contexts, endings):
        if option_list is None or any(e is None for e in option_list):
            continue
        first_sentences.extend([context] * len(option_list))
        second_sentences.extend(option_list)

    if not first_sentences or not second_sentences:
        return {}

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        padding="max_length",
        max_length=512
    )

    num_choices = len(endings[0])
    batch_size = len(contexts)

    input_ids = [tokenized["input_ids"][i * num_choices:(i + 1) * num_choices] for i in range(batch_size)]
    attention_mask = [tokenized["attention_mask"][i * num_choices:(i + 1) * num_choices] for i in range(batch_size)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Tokenize
tokenized_datasets = filtered_dataset.map(tokenize_function, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_dir="./logs",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

trainer.train()
