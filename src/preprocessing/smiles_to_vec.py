import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import ChemTokenizer, ChemBERTaModel
import torch

def smile_to_vector_RDKit(smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model that comes with RDKit 

    """
    mol = Chem.MolFromSmiles(smile)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Morgan fingerprint with radius 2
    vector = np.array(fp.ToBitString(), dtype=int)
    return vector

def smile_to_vector_ChemBERTa(model,tokenizer, smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model ChemBERTa

    I prefer this option over the RDKit method which produces a binary representation of the smile which need scaling
    """
    tokens = tokenizer.encode(smile, add_special_tokens=True)
    input_ids = torch.tensor([tokens])
    outputs = model(input_ids)
    vector = outputs[0].squeeze().detach().numpy()
    return vector





