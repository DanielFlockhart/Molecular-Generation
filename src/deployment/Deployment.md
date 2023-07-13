# Using the Trained Model for Molecular Generation

This markdown file explains the process of utilizing the newly trained model to generate new molecules based on a given potential molecule and additional conditions.

## Step 1: Model Loading

- Load the trained autoencoder model that has been trained on molecular skeleton image generation using the steps described in the training markdown file.

## Step 2: Preprocessing Inputs

- Recieve a starting point molecule from the user in the form of a Smile or the Name of a compound
- Vectorize the SMILES representation using either ChemBERTa or RDKit, and concatenate it with the condition vectors also provided by the user for the given molecule.
- The Condition vectors may control the generation of certain characteristics of the molecule

## Step 3: Encoding the Inputs

- Pass the concatenated vector representation through the encoder network of the trained model.
- Obtain the latent dimension vector, which captures the essential features of the potential molecule and the given conditions.

## Step 4: Generating New Molecules

- Concatenate the latent dimension vector obtained in the previous step with the condition vector for the desired new molecule.
- Pass this concatenated vector through the decoder network of the trained model.

## Step 5: Reconstruction and Post-processing

- Obtain the reconstructed representation from the decoder network, which represents the generated molecular skeleton for the new molecule.
- Post-process the generated molecular skeleton as required, such as converting it back to a SMILES representation using RDKit functions.

## Step 6: Analyzing and Selecting Generated Molecules

- Analyze the generated molecules based on desired criteria, such as drug-likeness, desired properties, or other constraints.
- Select the generated molecule(s) that satisfy the desired criteria.

By following these steps, you can use the trained model to generate new molecules based on a given potential molecule and additional conditions. The model leverages the learned patterns and structures from the training data to generate meaningful molecular structures that adhere to the provided inputs and conditions.

