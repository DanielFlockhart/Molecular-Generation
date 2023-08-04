# Generating/Mining Datasets for preprocessing

This markdown file explains the process of generating/gathering data from sources such as pubchem or chembl for the training process

## Step 1: Model Loading

## Create a Dataset of molecules and their features 

## Create Dataset of Illnesses and all scheduled/patented treatments of them



## Properties 
    Molecular Weight: Adjusting the molecular weight can affect the pharmacokinetics and duration of effect of the molecule. Larger molecules tend to have longer durations of action.

    XLogP (Partition Coefficient): Modifying the XLogP value can influence the molecule's hydrophobicity and its ability to cross cell membranes. Higher hydrophobicity might increase the duration of effect.

    TPSA (Topological Polar Surface Area): Changing the TPSA can impact the molecule's permeability and interaction with biological targets.

    H-bond Donor and Acceptor Count: Adjusting these counts can affect the molecule's ability to form hydrogen bonds with biological targets, potentially impacting the duration of effect.

    Rotatable Bond Count: Modifying the number of rotatable bonds can influence the molecule's flexibility and may affect its pharmacokinetics.

    Charge: Changing the charge of the molecule can impact its interaction with biological targets and affect the duration of action.

    Stereochemistry: Modifying atom and bond stereo counts can lead to different stereoisomers, which may exhibit different pharmacological properties, including duration and strength of effect.

    Conformer Count 3D: More conformers can increase the chances of finding a bioactive conformation that affects the molecule's interaction with its target.

    Feature Counts 3D: Modifying feature counts related to acceptors, donors, cations, anions, and hydrophobes can affect the molecule's interaction with its target and potential duration of effect.

    Covalent Unit Count: Changing the number of covalent units can lead to different formulations or modifications of the molecule, which might influence its properties.




By following these steps, you can generate datasets for you specific model.

