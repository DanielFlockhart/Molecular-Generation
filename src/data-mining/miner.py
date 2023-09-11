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
import concurrent.futures
import pandas as pd
class Miner:
    '''
    Class for mining data from PubChem.

    This class will create a dataset for the model to train on.
    '''
    def __init__(self):
        pass
    
    def search_compound(self,canonical_smiles):
        compounds = pcp.get_compounds(canonical_smiles, 'smiles')
        return compounds[0] if compounds[0].tpsa != None else None
    

    def get_compound(self, cid):
        return pcp.Compound.from_cid(cid)
    

    def get_molecular_properties(self,mol,keys):
        props_dict = {}
        for (i,prop) in enumerate(keys[2:]):
            try:
                attr = getattr(mol,prop)
                props_dict[prop] = attr
            except Exception as e:
                print(e)
                props_dict[prop] = "Unknown"
            
        return props_dict

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

            
    def add_entry(self,mol,keys,smile,name):
        props = self.get_molecular_properties(mol,keys)
        props['ID'] = name
        props['SMILES'] = smile

        df = pd.DataFrame(columns=keys)
        df.loc[len(df)] = props
        df.to_csv(file_constants.INPUTS_FOLDER, mode='a', header=False, index=False)

    def get_smiles(self,file):
        df = pd.read_csv(file)
        smiles = df['SMILE'].tolist()
        names = df['MOLECULE'].tolist()
        return smiles,names

keys =['ID','SMILES','atom_stereo_count','bond_stereo_count','charge','complexity','exact_mass','h_bond_acceptor_count','h_bond_donor_count','heavy_atom_count','molecular_weight','rotatable_bond_count','tpsa','xlogp']
if __name__ == '__main__':
    miner = Miner()
    file = file_constants.DATA_FOLDER+file_constants.DATASET+"/data.csv"
    smiles,names = miner.get_smiles(file)
    
    miner.create_database(keys)
    
    fail_count = 0
    for (i,mol) in tqdm(enumerate(smiles),total=len(smiles), bar_format=ui_constants.LOADING_BAR, ncols=80, colour='green'):
        try:
            compound = miner.search_compound(mol)
            if compound != None:
                miner.add_entry(compound,keys,mol,names[i])
            else:
                print("Error - No Molecule Found : ",mol)
        except Exception as e:
            fail_count += 1
            print("Error: ",mol)
            print(e)

    # Write any remaining data in the batch

    print("Failed: ",fail_count)
    print("Success: ",len(smiles)-fail_count)





'''
keys = ['Molecule', 'SMILES', 'Molecular Weight', 'TPSA', 'H-bond Donor Count', 'H-bond Acceptor Count', 'Rotatable Bond Count', 'Charge', 'Atom Stereo Count', 'Bond Stereo Count', 'Conformer Count 3D', 'Feature Acceptor Count 3D', 'Feature Donor Count 3D', 'Feature Cation Count 3D', 'Feature Ring Count 3D', 'Feature Hydrophobe Count 3D']
prop_dict = {"Molecule": mol.synonyms[0] if mol.synonyms else "Unknown"} if id==None else id # Add label for molecule name or SMILES
            prop_dict['SMILES'] = smiles
            mol = Chem.MolFromSmiles(smiles)

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
            


'''