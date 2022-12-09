import pandas as pd
from scipy.special import softmax
from sklearn import metrics
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_dataset

#MODEL = 'facebook/bart-base'
MODEL = 'distilbert-base-uncased'

dataset = load_dataset("hate_speech_offensive")['train']
train_testvalid = dataset.train_test_split(test_size=0.1)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

def label_preprocess_function(example):
    example['label'] = example['class']
    return example

def preprocess_function(examples):
    return tokenizer(examples["tweet"], truncation=True)

tokenized_data = train_testvalid.map(label_preprocess_function)
tokenized_data = tokenized_data.map(preprocess_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3)

training_args = TrainingArguments(
    output_dir="./models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    do_eval=True,
    do_predict=True,
    evaluation_strategy='epoch'
)
from sklearn.metrics import f1_score, roc_auc_score


def get_metrics(preds):
    y_true = preds.label_ids
    y_pred = preds.predictions.argmax(axis=1)
    print(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred, digits=3))
    return {
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'auc_macro': roc_auc_score(y_true, softmax(preds.predictions,axis=1), average='macro', multi_class='ovr'),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=get_metrics,
)

trainer.train()