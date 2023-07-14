from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
# Define the model repo
model_name = "seyonec/ChemBERTa-zinc-base-v1"

# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Transform input tokens
inputs = tokenizer("CC(=O)OC1=CC=CC=C1C(=O)O", return_tensors="pt")

# Model apply
outputs = model(**inputs)

# Extract the vector representation
vector1 = outputs.last_hidden_state.squeeze(0).mean(dim=0)



inputs = tokenizer("CC(=O)OC1=CC=CC=C1C(=O)C", return_tensors="pt")

# Model apply
outputs = model(**inputs)

# Extract the vector representation
vector2 = outputs.last_hidden_state.squeeze(0).mean(dim=0)


# Calculate the cosine similarity
cos = nn.CosineSimilarity(dim=0, eps=1e-6)
cos_sim = cos(vector1, vector2)
print(vector1.shape)
print(cos_sim.item())