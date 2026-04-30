import argparse
import os
import random
from datetime import datetime

import torch
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from peft import LoraConfig, TaskType, get_peft_model

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score
)

# =========================
# Argument Parser
# =========================
parser = argparse.ArgumentParser()

parser.add_argument("--train_csv", type=str, required=True)
parser.add_argument("--valid_csv", type=str, required=True)
parser.add_argument("--test_csv", type=str, required=True)

parser.add_argument("--model", type=str, required=True,
                    choices=["grover", "dnabert2", "nt"])

parser.add_argument("--aug_csv", type=str, default=None)
parser.add_argument("--output_dir", type=str, default="outputs")

args = parser.parse_args()

# =========================
# Seed
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Model Config
# =========================
if args.model == "grover":
    MODEL_NAME = "PoetschLab/GROVER"
    MAX_LEN = 1000
    CONFIG_TYPE = "auto"

elif args.model == "dnabert2":
    MODEL_NAME = "zhihan1996/DNABERT-2-117M"
    MAX_LEN = 510
    CONFIG_TYPE = "bert"

elif args.model == "nt":
    MODEL_NAME = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    MAX_LEN = 1000
    CONFIG_TYPE = "auto"

print("Model:", args.model)
print("HF model:", MODEL_NAME)

# =========================
# Load Data
# =========================
train_df = pd.read_csv(args.train_csv)
valid_df = pd.read_csv(args.valid_csv)
test_df = pd.read_csv(args.test_csv)

if args.aug_csv:
    aug_df = pd.read_csv(args.aug_csv)
    train_df = pd.concat([train_df, aug_df])
    print("Using augmentation")

# =========================
# Preprocess
# =========================
SEQ_COL = "DNA Sequence"
TARGET = "Drug Class"

for df in [train_df, valid_df, test_df]:
    df[SEQ_COL] = df[SEQ_COL].str.upper()
    df[SEQ_COL] = df[SEQ_COL].apply(lambda x: x[:MAX_LEN])

# =========================
# Label Mapping
# =========================
labels = sorted(train_df[TARGET].unique())
label_to_id = {l: i for i, l in enumerate(labels)}
id_to_label = {i: l for l, i in label_to_id.items()}

for df in [train_df, valid_df, test_df]:
    df["labels"] = df[TARGET].map(label_to_id)

# =========================
# Dataset
# =========================
def make_ds(df):
    return Dataset.from_dict({
        "data": df[SEQ_COL].tolist(),
        "labels": df["labels"].tolist()
    })

ds_train = make_ds(train_df)
ds_valid = make_ds(valid_df)
ds_test = make_ds(test_df)

# =========================
# Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(examples["data"], truncation=True)

ds_train = ds_train.map(tokenize, batched=True, remove_columns=["data"])
ds_valid = ds_valid.map(tokenize, batched=True, remove_columns=["data"])
ds_test = ds_test.map(tokenize, batched=True, remove_columns=["data"])

# =========================
# Model
# =========================
num_labels = len(label_to_id)

if CONFIG_TYPE == "bert":
    config = BertConfig.from_pretrained(MODEL_NAME)
else:
    config = AutoConfig.from_pretrained(MODEL_NAME)

config.num_labels = num_labels

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    config=config
)

model.to(device)

# =========================
# LoRA
# =========================
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=1,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =========================
# Training
# =========================
def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels = eval_pred.label_ids

    return {
        "f1_macro": f1_score(labels, preds, average="macro")
    }

training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=64,
    num_train_epochs=2,
    max_steps=1000,
    logging_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    label_names=["labels"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# =========================
# Evaluation
# =========================
collator = DataCollatorWithPadding(tokenizer)
loader = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=collator)

model.eval()

preds, trues = [], []

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k:v for k,v in batch.items() if k!="labels"}).logits
        p = torch.argmax(logits, dim=-1)

        preds.extend(p.cpu().numpy())
        trues.extend(batch["labels"].cpu().numpy())

print("Accuracy:", accuracy_score(trues, preds))
print("F1:", f1_score(trues, preds, average="macro"))
print("Balanced Acc:", balanced_accuracy_score(trues, preds))
print("Precision:", precision_score(trues, preds, average="macro"))
print("Recall:", recall_score(trues, preds, average="macro"))
