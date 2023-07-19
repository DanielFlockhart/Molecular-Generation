from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
# Define the model repo
model_name = "seyonec/ChemBERTa-zinc-base-v1"

# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Transform input tokens
inputs = tokenizer("COC1=C(C(OC1=O)c1ccccc1Cl)C(C)=NN=C(C)C1=C(OC)C(=O)OC1c1ccccc1Cl", return_tensors="pt")

# Model apply
outputs = model(**inputs)

# Extract the vector representation
vector1 = outputs.last_hidden_state.squeeze(0).mean(dim=0)

print(vector1)