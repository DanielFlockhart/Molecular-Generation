import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer,logging
import warnings,torch
logging.set_verbosity_error()
def smile_to_vector_RDKit(smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model that comes with RDKit 

    """
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Morgan fingerprint with radius 2
    vector = np.array(fp.ToBitString(), dtype=int)
    return vector.tolist()

def smile_to_vector_ChemBERTa(model_name, smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model ChemBERTa

    I prefer this option over the RDKit method which produces a binary representation of the smile which need scaling
    
    Unfortunately this does add alot to the processing time
    """
    # Use GPU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using {device} for ChemBERTa")
    # I have not got CUDA on my machine so I am using CPU
    # Download pytorch model
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Transform input tokens
    inputs = tokenizer(smile, return_tensors="pt").to(device)

    # Model apply
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the vector representation
    vector = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu()

    return vector.tolist()




import torch
from transformers import AutoModel, AutoTokenizer

def vector_to_smile_ChemBERTa(model_name, vector):
    """
    This function takes a ChemBERTa vector and returns the corresponding SMILES string.

    It uses the pretrained model ChemBERTa and its tokenizer.

    Note: The reconstructed SMILES string may not be an exact replica of the original input SMILES due to information loss during vectorization.

    Parameters:
    - model_name (str): Name or path of the pretrained ChemBERTa model.
    - vector (list): The ChemBERTa vector representing the SMILES.

    Returns:
    - str: The reconstructed SMILES string.
    """
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download pytorch model
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Convert the vector back to tensor
    vector_tensor = torch.tensor(vector).unsqueeze(0).to(device)

    # Model apply
    with torch.no_grad():
        outputs = model(inputs_embeds=vector_tensor)

    # Reconstruct the SMILES using the tokenizer
    reconstructed_smile = tokenizer.decode(outputs.input_ids[0], skip_special_tokens=True)

    return reconstructed_smile
