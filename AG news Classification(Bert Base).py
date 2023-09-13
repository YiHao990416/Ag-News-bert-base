# Importing the libraries needed
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig,BertModel
import numpy as np
     
from torch import cuda

#Identify and use the GPU
if torch.cuda.is_available():    

    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

# Import the csv into pandas dataframe 
df_train = pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
print(df_train.head(5))
print(df_test.head(5))

#Print the test and train dataset info and description
print(f"\nThe training dataframe info")
print(df_train.info())
print(f"\nThe training dataframe description")
print(df_train.describe())
print(f"\nThe testing dataframe info")
print(df_test.info())
print(f"\nThe testing dataframe description")
print(df_test.describe())

#Inspect any missing data. No missing data found
print(f"\nInspect any missing data from train data\n{df_train.isnull().sum()}")
print(f"\nInspect any missing data from test data\n{df_test.isnull().sum()}")
print(f"\nThe shape of train dataframe is {df_train.shape}")
print(f"\nThe shape of test dataframe is {df_test.shape}")

#Combine title and description into one column
df_train["Text"]=df_train[['Title','Description']].agg('. '.join, axis=1)  ## aggregate two column, separated by ". ", apply to all row(axis=1) 
df_train=df_train.drop(['Title','Description'],axis=1)

df_test["Text"]=df_test[['Title','Description']].agg('. '.join, axis=1)  ## aggregate two column, separated by ". ", apply to all row(axis=1) 
df_test=df_test.drop(['Title','Description'],axis=1)

#Convert the label to correct format
label_map={1:0,2:1,3:2,4:3}
df_train['Class Index'] = df_train['Class Index'].apply(lambda x : label_map[x])
df_test['Class Index'] = df_test['Class Index'].apply(lambda x : label_map[x])

print(df_train.head(10))
print(df_test.head(10))

#Define the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#Dataset and dataloader
class dataframe(Dataset):
    def __init__(self,x,y,tokenizer,max_len,transform=None):

        self.x=x  
        self.y=y.to_numpy()
        
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.transform=transform
  
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        x=str(self.x[index])
        y=self.y[index]

        encoding=self.tokenizer.encode_plus(
            x,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=True, 
            padding="max_length",
            truncation = True,
            return_attention_mask=True, 
            return_tensors='pt'
        )
        return {"input_ids":encoding['input_ids'].flatten(), "attention_mask":encoding['attention_mask'].flatten(), "target":torch.tensor(y,dtype=torch.long)}

#Define parameter for dataset and dataloader
batch_size_train= 64
batch_size_test=64
max_len_train=80
max_len_test=80

dataset_train=dataframe(x=df_train["Text"],y=df_train["Class Index"],tokenizer=tokenizer,max_len=max_len_train,transform=None)
dataset_test=dataframe(x=df_test["Text"],y=df_test["Class Index"],tokenizer=tokenizer,max_len=max_len_test,transform=None)

dataloader_train=DataLoader(dataset_train,batch_size=batch_size_train,shuffle=True,num_workers = 2)
dataloader_test=DataLoader(dataset_test,batch_size=batch_size_train,shuffle=True,num_workers = 2)

#Sentiment Classifier
class SentimentClassifier(torch.nn.Module):
    
    
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.drop = torch.nn.Dropout(p=0.45)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        layer = self.bert(input_ids = input_ids,attention_mask = attention_mask
        )
        
        output = self.drop(layer.get('pooler_output'))
        
        return self.out(output)

#Move the model to GPU
model=SentimentClassifier(4)
model=model.to(device)

#Define function used for training the model
loss_function=torch.nn.CrossEntropyLoss().to(device)
optimizer=torch.optim.AdamW(params=model.parameters(),lr=2e-5)   #Modify to optimize the training 
total_training_samples=len(dataset_train)
total_testing_samples=len(dataset_test)
epochs= 1                                                        #Modify to optimize the training 

print("------------------------Training initialized-------------------------------")

#train the model
for i in range(epochs):
    model.train()
    correct_predictions = 0
    j=0        
    for batch in dataloader_train:
    
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['target'].to(device)
        Final_input={'input_ids':input_ids,'attention_mask': attention_mask}
        optimizer.zero_grad()
        outputs=model(**Final_input)
        predicted_label=outputs.argmax(dim=1)
        loss = loss_function(outputs, target)   
        correct_predictions += ((predicted_label == target.view(-1)).sum()).item()
        loss.backward()                                        
        optimizer.step()
        j+=1
        print(f"{j} iterations,correct predictions is {correct_predictions}, accuracy is {(correct_predictions/(j*batch_size_train))*100}% ")

    print(f"Epoch {i}:the accuracy is {(correct_predictions/total_training_samples)*100}%")

#evaluate the model with test dataset    
for i in range(epochs):
    model.eval()
    correct_predictions = 0
    j=0
    with torch.no_grad():
        for batch in dataloader_test:
        
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            Final_input={'input_ids':input_ids,'attention_mask': attention_mask}
            outputs=model(**Final_input)
            predicted_label = outputs.argmax(dim=1)    ##return the max tensor position in each tensor array
            correct_predictions += ((predicted_label == target.view(-1)).sum()).item()   ## check the predicted= given label and sum up the correct preedictions, the view() is used to reshape the tensor array                                      ## .item() move the number to cpu
            j+=1

            print(f"{j} iterations,correct predictions is {correct_predictions}, accuracy is {(correct_predictions/(j*batch_size_test))*100}% ")

    print(f"Epoch {i}:the accuracy is {(correct_predictions/total_testing_samples)*100}%")
        
        


    
    



    

    
        


    
    



    
