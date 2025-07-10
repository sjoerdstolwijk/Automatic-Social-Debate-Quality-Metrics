# Import statements
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Added import
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from transformers import Trainer, TrainingArguments
from typing import Tuple
from transformers import EarlyStoppingCallback, IntervalStrategy
import random
import os
import shutil
import torch.nn as nn
from collections import Counter
from sklearn.metrics import f1_score, precision_recall_fscore_support

# Rest of the code remains unchanged.

# Set random seed and device
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define DATAPATH
DATAPATH = 'data/'

# Define test variables
#test_variables = ['HAS_OPINION_DUMMY', 'LIBERAL_DUMMY', 'CONSERVATIVE_DUMMY',  'HATELIST_FOCUSED_DUMMY', 'INTERACTIVITY_DUMMY', 'INCIVILITY_DUMMY', 'RATIONALITY_DUMMY']
#test_variables = ['HAS_OPINION_DUMMY', 'INCIVILITY_DUMMY', 'LIBERAL_DUMMY', 'CONSERVATIVE_DUMMY']
#test_variables =  ['INCIVILITY_DUMMY', 'CONSERVATIVE_DUMMY' ,  'LIBERAL_DUMMY', 'HAS_OPINION_DUMMY']
test_variables =  ['LIBERAL_DUMMY', 'HAS_OPINION_DUMMY']


# Read train and test sets
train_set = pd.read_csv(f'{DATAPATH}train.csv')[test_variables + ['ID'] + ['commentText']]
test_set = pd.read_csv(f'{DATAPATH}test.csv')[test_variables + ['ID'] + ['commentText']]

# Define My_Dataset class
class My_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Define create_datasets function
def create_datasets(concept, train_set, test_set, tokenizer, max_length=512, test_size=.2, random_state=42, downsample_majority=True):
    train_labels = train_set[concept]
    test_labels =  test_set[concept]
    train_texts = train_set['commentText'] 
    test_texts =  test_set['commentText']
    
    if downsample_majority and concept in ['INCIVILITY_DUMMY', 'LIBERAL_DUMMY', 'HAS_OPINION_DUMMY']:
       # Identify majority and minority classes
        majority_class = train_labels.value_counts().idxmax()
        minority_class = train_labels.value_counts().idxmin()

        # Calculate the number of instances in the minority class
        minority_count = len(train_labels[train_labels == minority_class])

        # Sample an equal number of instances from the majority class
        majority_indices = train_labels[train_labels == majority_class].index
        random_indices_to_keep = np.random.choice(majority_indices, size=minority_count, replace=False)

        # Concatenate instances from both classes
        minority_indices = train_labels[train_labels == minority_class].index
        random_indices_to_keep = np.concatenate((random_indices_to_keep, minority_indices))

        # Shuffle the indices to mix both classes
        np.random.shuffle(random_indices_to_keep)

        # Update train_labels and train_texts
        train_labels = train_labels[random_indices_to_keep]
        train_texts = train_texts[random_indices_to_keep]
        
    elif downsample_majority and concept in ['CONSERVATIVE_DUMMY']:
    # Identify majority and minority classes
        majority_class = train_labels.value_counts().idxmax()
        minority_class = train_labels.value_counts().idxmin()

        # Calculate the number of instances in the minority class
        minority_count = len(train_labels[train_labels == minority_class])

        # Calculate the number of instances needed to achieve a 2/5 minority class
        target_minority_count = int((3 / 5) * len(train_labels))

        # Sample instances from the majority class to reach the target minority count
        majority_indices = train_labels[train_labels == majority_class].index
        random_indices_to_keep_majority = np.random.choice(majority_indices, size=target_minority_count - minority_count, replace=False)

        # Concatenate instances from both classes
        minority_indices = train_labels[train_labels == minority_class].index
        random_indices_to_keep_minority = minority_indices

        # Combine the sampled majority indices and minority indices
        random_indices_to_keep = np.concatenate((random_indices_to_keep_majority, random_indices_to_keep_minority))

        # Shuffle the indices to mix both classes
        np.random.shuffle(random_indices_to_keep)

        # Update train_labels and train_texts
        train_labels = train_labels[random_indices_to_keep]
        train_texts = train_texts[random_indices_to_keep]

    class_weights = (1 - (test_set[concept].value_counts().sort_index() / len(train_set))).values
    class_weights = torch.from_numpy(class_weights).float().to("cuda")

    train_texts, val_texts, train_labels, val_labels = train_test_split(list(train_texts), list(train_labels), test_size=test_size, random_state=random_state)

    print(concept)
    print(Counter(train_labels))
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings  = tokenizer(list(test_texts), truncation=True, padding=True, max_length=max_length)

    train_dataset = My_Dataset(train_encodings, train_labels)
    val_dataset = My_Dataset(val_encodings, val_labels)
    test_dataset = My_Dataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset, train_texts, val_texts, train_labels, val_labels, class_weights


# Define compute_minority_f1 function
def compute_minority_f1(y_true, y_pred):
    minority_class = np.unique(y_true)[np.argmin(np.bincount(y_true))]
    minority_indices = np.where(y_true == minority_class)[0]
    minority_true = y_true[minority_indices]
    minority_pred = y_pred[minority_indices]

    return f1_score(minority_true, minority_pred)


def compute_weighted_f1(y_true, y_pred):
    f1_scores = f1_score(y_true, y_pred, average=None)
    class_counts = np.bincount(y_true)
    weighted_f1 = np.sum(f1_scores * class_counts) / len(y_true)
    return weighted_f1

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    acc = accuracy_score(labels, preds)
    
    macro_f1 = f1_score(labels, preds, average='macro')
    minority_f1 = compute_minority_f1(labels, preds)
    
    weighted_f1 = compute_weighted_f1(labels, preds)
    
    return {'macro_f1': macro_f1, 'minority_f1' : minority_f1, 'weighted_f1': weighted_f1, 'acc': acc}


# Define WeightedLossTrainer class
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss
    


def train_and_find_best_hyperparams(concept, model, train_dataset, val_dataset, compute_metrics, learning_rates, batch_sizes, class_weights) -> Tuple[str, dict]:
    best_f1 = 0
    best_report = ""
    best_hyperparams = ""
    best_model_path = ""

    print(class_weights)

    for lr in learning_rates:
        torch.cuda.empty_cache()

        for bs in batch_sizes:

            torch.cuda.empty_cache()
            
            if concept in ['INCIVILITY_DUMMY', 'HAS_OPINION_DUMMY', 'CONSERVATIVE_DUMMY', 'LIBERAL_DUMMY']:
              
            
           # if concept == 'HAS_OPINION_DUMMY' or concept == 'INCIVILITY_DUMMY':
                metric_name = 'weighted_f1'
                print(metric_name)
                          
            else: 
                metric_name = "minority_f1"  # Use minority class F1-score as the metric
                print(metric_name)

            for warmup_steps in [0, 100]:

                random_seed = 42
                np.random.seed(random_seed)
                torch.manual_seed(random_seed)
                torch.cuda.manual_seed(random_seed)
                torch.cuda.manual_seed_all(random_seed)

                training_args = TrainingArguments(
                    num_train_epochs=20,
                    per_device_train_batch_size=bs,
                    per_device_eval_batch_size=bs,
                    learning_rate=lr,
                    load_best_model_at_end=True,
                    metric_for_best_model=metric_name,
                    warmup_steps=warmup_steps,
                    weight_decay=0.01,
                    output_dir='../../../data/volume_2/publicsphere-new-new',
                    logging_dir='./logs',
                    logging_steps=50,
                    evaluation_strategy='steps',
                    gradient_accumulation_steps = 8,
                    gradient_checkpointing=True,
                    fp16=True,
                    save_total_limit=3 
                )

                early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10)  # Adjust patience
                
                
                #if concept == 'HAS_OPINION_DUMMY' or concept == 'INCIVILITY_DUMMY':
                    
                if concept in ['INCIVILITY_DUMMY', 'CONSERVATIVE_DUMMY', 'LIBERAL_DUMMY', 'HAS_OPINION_DUMMY' , 'INCIVILITY_DUMMY']:
                    print("Continue with regular Trainer")

                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=compute_metrics,
                        callbacks=[early_stopping_callback])
                        
                else:
                
                    print("Initialize WeightedLossTrainer")
                    
                    trainer = WeightedLossTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=val_dataset,
                        compute_metrics=compute_metrics,
                        callbacks=[early_stopping_callback])

                print('start training with: \n\n')
                print(f'Learning rate: {lr}')
                print(f'batch size: {bs}')
                print(f'Warmup steps: {warmup_steps}')

                trainer.train()

                evaluation_metrics = trainer.evaluate(eval_dataset=val_dataset)
                f1 = evaluation_metrics['eval_macro_f1']  # Use the minority class F1-score

                print(f"{concept}")
                print(f"current metric: {f1}")
                print(f"best metric: {best_f1}")

                if f1 > best_f1:
                    if os.path.exists(best_model_path):
                        shutil.rmtree(best_model_path)

                    best_f1 = f1
                    predicted_val = trainer.predict(val_dataset)
                    predicted_val_labels = predicted_val.predictions.argmax(-1)
                    predicted_val_labels = predicted_val_labels.flatten().tolist()
                    print(f'*************BEST MODEL {concept}*************')
                    print(classification_report(val_labels, predicted_val_labels))
                    best_report = classification_report(val_labels, predicted_val_labels, output_dict=False)
                    best_hyperparams = {"learning_rate": lr, "batch_size": bs, "concept" : concept, "warmup_steps": warmup_steps}
                    best_model_path = f'../../../data/volume_2/publicsphere-new-new/best_f1_model_{concept}_{bs}_{lr}_{warmup_steps}'
                    trainer.save_model(best_model_path)
                    print(f'**************MODEL SAVED************************')

    return best_report, best_hyperparams


# Define model_name and device_name
model_name = 'bert-base-uncased'
device_name = 'cuda'
max_length = 512
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device_name)

# Define hyperparameters
#learning_rates = [1e-4, 2e-4, 3e-4]
learning_rates = [3e-6, 3e-5, 1e-2]
#learning_rates = [3e-9, 3e-7]
batch_sizes = [10]

#concepts = ['HAS_OPINION_DUMMY', 'LIBERAL_DUMMY', 'CONSERVATIVE_DUMMY',  'HATELIST_FOCUSED_DUMMY', 'INTERACTIVITY_DUMMY', 'INCIVILITY_DUMMY', 'RATIONALITY_DUMMY']

concepts = ['LIBERAL_DUMMY', 'HAS_OPINION_DUMMY']

all_best_reports = []
all_best_params = []

for concept in concepts:
    print(f'***********************{concept}**********************\n\n')
    

    train_dataset, val_dataset, test_dataset, train_texts, val_texts, train_labels, val_labels, class_weights = create_datasets(concept, train_set, test_set, tokenizer, max_length=512, test_size=.2, random_state=42)
    best_report, best_params = train_and_find_best_hyperparams(concept, model, train_dataset, val_dataset, compute_metrics, learning_rates, batch_sizes, class_weights) 

    # Append results to lists
    all_best_reports.append(best_report)
    all_best_params.append(best_params)

    # Save the objects to disk after each concept iteration
    with open(f'all_best_reports_{concept}.pkl', 'wb') as f:
        pickle.dump(all_best_reports, f)
    with open(f'all_best_params_{concept}.pkl', 'wb') as f:
        pickle.dump(all_best_params, f)

# Define the list of concepts