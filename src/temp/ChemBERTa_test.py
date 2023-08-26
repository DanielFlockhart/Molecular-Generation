'''
Tests for the ChemBERTa model to see how it generates vectors from SMILES strings
'''


from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
# Define the model repo
model_name = "seyonec/ChemBERTa-zinc-base-v1"

# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Transform input tokens
inputs = tokenizer("COc1ccc(OC)c2sc(NC(=O)c3cc4c(ccc5ccccc54)oc3=O)nc12", return_tensors="pt")
# Model apply
outputs = model(**inputs)

# Extract the vector representation
vector1 = outputs.last_hidden_state.squeeze(0).mean(dim=0)

print(vector1)