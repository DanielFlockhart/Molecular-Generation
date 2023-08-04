import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import Draw
from PIL import Image
import sys, os
sys.path.insert(0, os.path.abspath('..'))

from Constants import file_constants
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


    def get_compound(self, cid):
        return pcp.Compound.from_cid(cid)
    

    def get_molecular_properties(self,molecules):
        properties = []
        for mol in molecules:
            

            if mol is not None:
                prop_dict = {"Molecule": mol.synonyms[0] if mol.synonyms else "Unknown"}  # Add label for molecule name or SMILES
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
                
                # Calculate Covalent Unit Count (Assuming the molecule is a single covalent unit)
                prop_dict['Covalent Unit Count'] = 1
                
                properties.append(prop_dict)
        return properties


    def create_database(self):
        '''
        Generates Dataset
        '''


        print(format_title("Getting Vector Representations of Smiles and Conditions"))
        self.smiles = self.database.get_smiles_from_ids()

        # Clear Inputs Folder
        file_utils.clear_csv(file_constants.INPUTS_FOLDER)
        # Add Headers
        df = pd.DataFrame(columns=['ID', 'SMILES'])
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=True, index=False)
        print("Warning: This may take a while")
        print("Dataset must be sorted by ID alphabetically, will make it more robust later")

        #target_vecs,labels = img_utils.load_images() # This line causes 30Gb of RAM to be used

        for (i,smile) in tqdm(enumerate(self.smiles),total=len(self.smiles), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
            
            id = self.database.get_id(smile)
            #print("Warning, do not load target from inputs.csv as it is not the full image")
            
            (smiles_vec, condition_vec) = self.get_input(smile)
            self.add_entry(id,smile,condition_vec,smiles_vec)
            
    def add_entry(self,id):

        # Create an empty DataFrame
        
        df = pd.DataFrame(columns=['ID', 'SMILES', 'conditions', 'vector'])
        # Create a new row as a list
        new_row = [id, smile, conditions, smiles_vec]
        # Append the new row to the DataFrame
        df.loc[len(df)] = new_row

        # Write the DataFrame to a CSV file
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=False, index=False)

    

if __name__ == '__main__':
    miner = Miner()
    compounds = [miner.get_compound(1615+i) for i in range(10)]
    props = miner.get_molecular_properties(compounds)
    print(props)

kwargs = ["Mol_Weight", "TPSA", "H-bond_Dons", "H-bond_Accs", "Rotatable_Bonds", "Charge", "Atom_Stereos", "Bond_Stereos", "Conformers_3D", "Feature_Accs_3D", "Feature_Dons_3D", "Feature Cation Count 3D", "Feature Ring Count 3D", "Feature Hydrophobe Count 3D", "Covalent Unit Count"]
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