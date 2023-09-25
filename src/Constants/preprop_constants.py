# Preprocessing Constants
IMG_SIZE = 400 # Size to save the images in the dataset
STD_DEV = 2.2 # Standard deviation of scaling factor for the images - Higher = Images will have larger range of sizes
MAX_CHARS = 250 # Maximum number of characters of smile used for file name
BOUND_PERCENTILE = 98.5
SUBSET_COUNT = 100
EMBEDDING_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
keys =['atom_stereo_count','complexity','exact_mass','h_bond_acceptor_count','heavy_atom_count','molecular_weight','rotatable_bond_count','tpsa','xlogp']
