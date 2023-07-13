import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from transformers import ChemTokenizer, ChemBERTaModel
import torch

def smile_to_vector_RDKit(smile):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model that comes with RDKit 

    """
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)  # Morgan fingerprint with radius 2
    vector = np.array(fp.ToBitString(), dtype=int)
    return vector

def smile_to_vector_ChemBERTa(smile,other_data):
    """
    This function takes a smile string and returns a vector from a pretrained word embedding it uses the pretrained model ChemBERTa
    """
    tokens = tokenizer.encode(smiles, add_special_tokens=True)
    input_ids = torch.tensor([tokens])
    outputs = model(input_ids)
    vector = outputs[0].squeeze().detach().numpy()
    return vector


def conditions_to_vector(conditions):
    '''
    Convert Conditions for new molecule
    ''' 


smiles = 'CC(=O)Oc1ccccc1C(=O)O'

rdkit_vector = smile_to_vector_RDKit(smiles)
print("RD Kit Created Vector")
print(rdkit_vector)




# Assuming you have the pre-trained ChemBERTa model and tokenizer
model = ChemBERTaModel.from_pretrained('path_to_pretrained_model')
tokenizer = ChemTokenizer.from_pretrained('path_to_tokenizer')

chemberta_vector = smile_to_vector_ChemBERTa(smiles)
print("ChemBERTa Created Vector")
print(chemberta_vector)