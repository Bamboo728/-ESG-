# -*- coding: utf-8 -*-
import pandas as pd
import json
import numpy as np
import pickle


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset


with open('D:\Pycharm\Project\learning\Graduation_F\E_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 标签编码为数字
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

train_df, val_df = train_test_split(df, test_size=7, stratify=df["label"], random_state=42)


model_path = r"D:\Anaconda\models\text2vec-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)



def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["text", "label"]])

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


num_labels = len(label_encoder.classes_)
model = BertForSequenceClassification.from_pretrained(
    model_path, local_files_only=True, num_labels=num_labels
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


training_args = TrainingArguments(
    output_dir="./E_model",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()


model.save_pretrained("./E_model")
tokenizer.save_pretrained("./E_model")

with open("./E_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


