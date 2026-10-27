from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

import torch


import pandas as pd

from sklearn.metrics import classification_report

import torch
from sklearn.model_selection import train_test_split

import glob


# specify the directory path
dir_path = 'models/'
DATAPATH = 'data/'
# use glob to get the list of folders


test_variables = ['INTERACTIVITY_DUMMY', 'INCIVILITY_DUMMY', 'HATELIST_FOCUSED_DUMMY', 'RATIONALITY_DUMMY', 'HAS_OPINION_DUMMY', 'LIBERAL_DUMMY', 'CONSERVATIVE_DUMMY']

train_set = pd.read_csv(f'{DATAPATH}train.csv')[test_variables+['ID']+['commentText']]  
test_set = pd.read_csv(f'{DATAPATH}test.csv')[test_variables+['ID']+['commentText']]  

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
    
def create_datasets(concept, train_set, test_set, tokenizer, max_length=512, test_size=.2, random_state=42):
    
    train_labels = train_set[concept]
    test_labels =  test_set[concept]
    train_texts = train_set['commentText'] 
    test_texts =  test_set['commentText']
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(list(train_texts), list(train_labels), test_size=test_size, random_state=random_state)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings  = tokenizer(list(test_texts), truncation=True, padding=True, max_length=max_length)

    train_dataset = My_Dataset(train_encodings, train_labels)
    val_dataset = My_Dataset(val_encodings, val_labels)
    test_dataset = My_Dataset(test_encodings, test_labels)

    return train_dataset, val_dataset, test_dataset, train_texts, val_texts, test_texts, train_labels, val_labels, test_labels



def get_class_report(CONCEPT):
    file = glob.glob(f'{dir_path}/*{CONCEPT}*')[0]

    train_dataset, val_dataset, test_dataset, train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = create_datasets(CONCEPT, train_set, test_set, tokenizer, max_length=512, test_size=.2, random_state=42)
    model = AutoModelForSequenceClassification.from_pretrained(file, num_labels=2)
    
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True, device=0)

    predicted_test_scores = pipe(list(test_texts), truncation=True)
    predicted_test_labels = [model.config.label2id[prediction['label']] for prediction in predicted_test_scores]

    print(classification_report(test_labels, predicted_test_labels))    

    df = pd.DataFrame({'commentText': test_set['commentText'], 'ID': test_set['ID'], 'Text': test_texts, 'Predicted_Label': predicted_test_labels, 'True_Label': test_labels})
    return classification_report(test_labels, predicted_test_labels, output_dict=True), df

measure_variables = ['INTERACTIVITY_DUMMY', 'INCIVILITY_DUMMY', 'HATELIST_FOCUSED_DUMMY', 'RATIONALITY_DUMMY', 'HAS_OPINION_DUMMY', 'LIBERAL_DUMMY', 'CONSERVATIVE_DUMMY']

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

performance = []

classif = ['0', '1', 'macro avg', 'weighted avg']

all_test_dfs = []

for label in measure_variables:
    print(label)
    
    classfication, test_df = get_class_report(label)
    test_df[label] = test_df['Predicted_Label']

    
    for c in classif:
        performance.append({'Dimension': label, 'Class':c,'Precision':classfication[c]['precision'],
                                            'Recall':classfication[c]['recall'],'F1-score':classfication[c]['f1-score'],'support':classfication[c]['support'],'Accuracy_overall':classfication['accuracy']}) 
    # all_test_dfs.append(test_df[['commentText', 'ID', 'Text', "label[1]"]])
    
performance_df = pd.DataFrame(performance)
performance_df.to_csv("models_og_script_retrained.csv")
print(performance_df)