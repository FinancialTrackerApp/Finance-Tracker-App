#!/usr/bin/env python
# coding: utf-8

# In[77]:


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os


# In[78]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




# In[79]:


# csv_path = r"C:\Users\final\FinancialTrackerApp\model_code\expenses_data.csv"
csv_path = r"D:\Programming\Projects\Finance-Tracker-App\python_stuffs\model_code\expenses_data_cleaned.csv"
# BASE_DIR = os.path.join(os.getcwd(), "model_code")
# csv_path = os.path.join(BASE_DIR, "expenses_data.csv")
df = pd.read_csv(csv_path)
texts = df['text']       
labels = df['category']  


# In[80]:


#converting text data into numerical vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
X.shape


# In[81]:


# Encoding the labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)
y


# In[82]:


#Train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert sparse matrices to dense arrays to avoid type errors
X_train = X_train.toarray()
X_test = X_test.toarray()


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# In[83]:


X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)


# In[84]:


#Define model
class ExpenseClassifier(nn.Module):
    def __init__(self, input_size,hidden_size, num_classes):
        super(ExpenseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# In[85]:


X_train.shape[1]


# In[86]:


input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(set(y))

model = ExpenseClassifier(input_size, hidden_size, num_classes)
model.to(device)


# In[87]:


#Loss and optimizer
criterion = nn.CrossEntropyLoss() # compares predicted catgeory vs actual
optimizer = optim.Adam(model.parameters(), lr=0.01) # updates model weights efficiently


# In[88]:


#Training loop
epochs = 200
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%10==0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# In[89]:


#Evaluation
with torch.inference_mode():
    y_pred = model(X_test) #forward pass
    y_pred_classes = torch.argmax(y_pred, dim=1 ) #pick highest probability class
    acc = (y_pred_classes == y_test).float().mean() #accuracy
    print(f'Accuracy: {acc.item():.4f}')


# In[90]:


import torch
import joblib
import os

# Assume you already trained
# vectorizer = TfidfVectorizer().fit(texts)
# model = ExpenseClassifier(input_size=..., hidden_size=..., output_size=...) ; model trained
# y_train = list of categories in training set (e.g., ["Transport", "Healthcare", ...])

save_dir = "pytorch_models"
os.makedirs(save_dir, exist_ok=True)

# Save TF-IDF vectorizer
joblib.dump(vectorizer, os.path.join(save_dir, "vectorizer.pkl"))

# Save model weights only
torch.save(model.state_dict(), os.path.join(save_dir, "category_predictor_model.pth"))

# Save category mapping (encoder) using PyTorch
category_list = ["Transport", "Healthcare", "Food", "Housing", "Education", "others"]
torch.save(category_list, os.path.join(save_dir, "encoder.pth"))

print("✅ Vectorizer, model weights, and encoder saved")


# In[91]:


import spacy
nlp = spacy.load("en_core_web_sm")
totals = {
    "Transport": 0.0,
    "Healthcare": 0.0,
    "Food": 0.0,
    "Housing": 0.0,
    "Education": 0.0,
    "others": 0.0
}

def predict(text):
    # Vectorize and move to correct device
    vec = vectorizer.transform([text]).toarray()
    vec = torch.tensor(vec, dtype=torch.float32).to(device)

    # Predict category
    output = model(vec)
    pred = torch.argmax(output, 1).item()
    pred = encoder.inverse_transform([pred])[0]

    # Extract all amounts
    doc = nlp(text)
    amount = 0.0
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "CARDINAL"]:
            try:
                amount += float(ent.text)  # sum all numbers in the sentence
            except:
                pass

    totals[pred] += amount
    # return totals
# testing:
predict("Bought apples for 80 rs")   
predict("Hospital bill as 750")   
predict("Taxi fare as 300")       
predict("Netflix subscription as 500") 
predict("Train from Velachery as 150") 
predict("Spent 500 at hotel")

print(totals)


# In[92]:


import spacy
nlp = spacy.load("en_core_web_sm")

doc = nlp("I spent Rs. 700 and 300 on food on 26th July")
for ent in doc.ents:
    print(ent.text, ent.label_)


# In[93]:


import os
print(os.getcwd())



# In[94]:


encoder_path = os.path.join("model_code", "pytorch_models", "encoder.pkl")
encoder = joblib.load(encoder_path)
print(encoder.classes_)



