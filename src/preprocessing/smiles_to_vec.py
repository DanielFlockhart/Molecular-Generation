import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import AutoModel, AutoTokenizer

def smile_to_vector_RDKit(smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model that comes with RDKit 

    """
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Morgan fingerprint with radius 2
    vector = np.array(fp.ToBitString(), dtype=int)
    return vector

def smile_to_vector_ChemBERTa(model_name, smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model ChemBERTa

    I prefer this option over the RDKit method which produces a binary representation of the smile which need scaling
    """
    # Download pytorch model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Transform input tokens
    inputs = tokenizer(smile, return_tensors="pt")

    # Model apply
    outputs = model(**inputs)

    # Extract the vector representation
    vector = outputs.last_hidden_state.squeeze(0).mean(dim=0)

    return vector





