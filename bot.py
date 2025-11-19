


#NoteBook For Colab ---

!pip install -q transformers datasets torch accelerate

from datasets import load_dataset

dataset = load_dataset("banking77")
dataset

dataset = dataset.rename_column("text", "sentence")
dataset = dataset.rename_column("label", "labels")

num_intents = len(set(dataset["train"]["labels"]))
num_intents

#-----

from transformers import AutoTokenizer
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

tokenized = dataset.map(tokenize, batched=True)

tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

#-------

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=num_intents
)

import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./banking77_model",
    logging_steps=20,
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    report_to="none"
)

#--------

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"]
)

trainer.train()

#--------

trainer.evaluate()

#--------

#Saving Model
model.save_pretrained("/content/banking77_intent_model")
tokenizer.save_pretrained("/content/banking77_intent_model")

#--------

#Uploading HuggingFace

from huggingface_hub import login

login("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")

from huggingface_hub import HfApi, create_repo

create_repo(
    repo_id="NauRaa/banking77-intent-model",  
    repo_type="model",
    exist_ok=True  
)

#--------------------------------