import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Draw
from PIL import Image
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from Constants import file_constants,ui_constants
from ui.terminal_ui import *
from utilities import file_utils
from tqdm import tqdm

import pandas as pd
class Miner:
    '''
    Class for mining data from PubChem.

    This class will create a dataset for the model to train on.
    '''
    def __init__(self):
        pass
    
    def search_compound(self,canonical_smiles):
        return pcp.get_compounds(canonical_smiles, 'smiles')[0]

    def get_compound(self, cid):
        return pcp.Compound.from_cid(cid)
    

    def get_molecular_properties(self,mol):
        
        if mol is not None:
            prop_dict = {"Molecule": mol.synonyms[0] if mol.synonyms else "Unknown"}  # Add label for molecule name or SMILES
            prop_dict['SMILES'] = mol.canonical_smiles
            mol = Chem.MolFromSmiles(mol.canonical_smiles)

            # Calculate molecular weight
            prop_dict['Molecular Weight'] = Descriptors.MolWt(mol)
            
            # Calculate TPSA
            prop_dict['TPSA'] = Descriptors.TPSA(mol)
            
            # Calculate H-bond Donor and Acceptor Count
            prop_dict['H-bond Donor Count'] = Descriptors.NumHDonors(mol)
            prop_dict['H-bond Acceptor Count'] = Descriptors.NumHAcceptors(mol)
            
            # Calculate Rotatable Bond Count
            prop_dict['Rotatable Bond Count'] = Descriptors.NumRotatableBonds(mol)
            
            # Calculate Charge (Note: You may need to have charge information in the molecule representation)
            prop_dict['Charge'] = 0  # Replace 0 with the charge value if available
            
            # Calculate Stereochemistry
            stereo_info = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
            prop_dict['Atom Stereo Count'] = len([atom for atom, _ in stereo_info if atom is not None])
            prop_dict['Bond Stereo Count'] = len(stereo_info) - prop_dict['Atom Stereo Count']            
            # Calculate Conformer Count 3D (using ETKDG method)
            mol = Chem.AddHs(mol)  # Add hydrogens for better conformer generation
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=100, params=AllChem.ETKDG())
            prop_dict['Conformer Count 3D'] = len(cids)
            
            # Calculate Feature Counts 3D
            prop_dict['Feature Acceptor Count 3D'] = Chem.rdMolDescriptors.CalcNumLipinskiHBA(mol)
            prop_dict['Feature Donor Count 3D'] = Chem.rdMolDescriptors.CalcNumLipinskiHBD(mol)
            prop_dict['Feature Cation Count 3D'] = abs(prop_dict['Charge'])
            prop_dict['Feature Ring Count 3D'] = Descriptors.RingCount(mol)
            prop_dict['Feature Hydrophobe Count 3D'] = Descriptors.NumAliphaticCarbocycles(mol) + Descriptors.NumAliphaticHeterocycles(mol) + Descriptors.NumAliphaticRings(mol)
            
        return prop_dict


    def create_database(self,keys):
        '''
        Generates Dataset
        '''

        print(format_title("Creating Dataset"))

        # Clear Inputs Folder
        file_utils.clear_csv(file_constants.INPUTS_FOLDER)

        # Add Headers
        df = pd.DataFrame(columns=keys)
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=True, index=False)

            
    def add_entry(self,mol):
        props = self.get_molecular_properties(mol)
        df = pd.DataFrame(columns=keys)
        df.loc[len(df)] = props
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=False, index=False)

    def get_smiles(self,file):
        df = pd.read_csv(file)
        smiles = df['SMILES'].tolist()
        return smiles

    
keys = ['Molecule', 'SMILES', 'Molecular Weight', 'TPSA', 'H-bond Donor Count', 'H-bond Acceptor Count', 'Rotatable Bond Count', 'Charge', 'Atom Stereo Count', 'Bond Stereo Count', 'Conformer Count 3D', 'Feature Acceptor Count 3D', 'Feature Donor Count 3D', 'Feature Cation Count 3D', 'Feature Ring Count 3D', 'Feature Hydrophobe Count 3D']
if __name__ == '__main__':
    miner = Miner()
    file = file_constants.DATA_FOLDER+"/db2-pas/names_and_smiles.csv"
    smiles = miner.get_smiles()
    
    miner.create_database(keys)
    for (i,mol) in tqdm(enumerate(smiles),total=len(smiles), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
        try:
            compound = miner.search_compound(mol)
            miner.add_entry(compound)
        except:
            print("Error: ",mol)



'''
    Molecular Weight

    TPSA

    H-bond Donor and Acceptor Count

    Rotatable Bond Count

    Charge

    Stereochemistry

    Conformer Count 3D

    Feature Counts 3D

    Covalent Unit Count
'''