import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json

import numpy
import pandas

import datasets
import transformers

from sklearn.metrics import classification_report


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate.big_modeling").setLevel(logging.ERROR)


SEED: int = 13


MODEL_NAME: str = "google-bert/bert-base-uncased"

TRAINER_XARGS = dict(
    # hyper-parameter
    learning_rate=2e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    # gpu device config
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    # input/output utility
    eval_strategy="no",
)

DATA_PATHS: dict[str, str] = {
    "train": "./data/train.csv",
    "test": "./data/test.csv",
}
SRC_COLUMN: str = "commentText"
TGT_COLUMNS: list[str] = [
     'INTERACTIVITY_DUMMY', 'INCIVILITY_DUMMY', 'HATELIST_FOCUSED_DUMMY', 
     'RATIONALITY_DUMMY', 'HAS_OPINION_DUMMY', 'LIBERAL_DUMMY', 'CONSERVATIVE_DUMMY'
]

DATASETS: dict[str, pandas.DataFrame] = {
    split: (
        pandas.read_csv(path)
        .sample(frac=1.0, random_state=SEED)
    ) for split, path in DATA_PATHS.items()
}

TOKENIZER = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(sample: str):
    return TOKENIZER(sample[SRC_COLUMN], truncation=True)


DATASETS_HF: dict[str, datasets.Dataset] = {
    split: (
         datasets.Dataset
         .from_pandas(df)
         .map(preprocess, batched=True)
         .rename_column(SRC_COLUMN, "text")
    )
    for split, df in DATASETS.items()
}


def compute_metrics(outputs):
    predictions, labels = outputs
    predictions = numpy.argmax(predictions, axis=1)

    return classification_report(
        predictions, labels,
        output_dict=True, zero_division=1.
    )

results: list[dict] = []

for tgt_annotator_column in TGT_COLUMNS:
    sim_model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    sim_data = {
        split: df.rename_column(tgt_annotator_column, "label")
        for split, df in DATASETS_HF.items()
    }

    trainer = transformers.Trainer(
        model=sim_model,
        args=transformers.TrainingArguments(**TRAINER_XARGS),
        train_dataset=sim_data["train"],
        data_collator=transformers.DataCollatorWithPadding(tokenizer=TOKENIZER),
        processing_class=TOKENIZER,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results.append({
        "construct": tgt_annotator_column,
        "embedding": "BERT (fine-tuned)",
        "classifier": "BERT Classifier Head (fine-tuned)",
    } | trainer.evaluate(sim_data["test"]))

    trainer.save_model(f"models/{tgt_annotator_column}")


json.dump(results, open("./data/new_train.results.json", "w"), indent=4)