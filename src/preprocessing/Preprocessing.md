# Preprocessing Data for Molecular Skeleton Image Generation

This markdown file outlines the preprocessing steps for generating molecular skeleton images from SMILES strings using ChemBERTa or RDKit, and further processing the targets to ensure consistent scaling.

## Step 1: SMILES to Vector Conversion

To represent SMILES strings as vector representations, we can use either ChemBERTa or RDKit.

- For the RDKit approach, we utilize the RDKit library to convert SMILES strings into vector representations using Morgan fingerprints. This generates a binary vector representation that encodes structural information of the molecule.

- For the ChemBERTa approach, we employ a pre-trained ChemBERTa model and tokenizer to convert SMILES strings into vector representations. The ChemBERTa model has been fine-tuned on chemical language and provides dense vector representations that capture semantic and contextual information.

## Step 2: Condition Information Vectorization and Concatenation

In addition to SMILES vectorization, we convert condition information about the molecule into a vector representation. This information can include various properties or characteristics of the molecule. The condition information is vectorized using an appropriate technique, such as one-hot encoding or word embeddings.

After vectorizing the condition information, it is concatenated with the SMILES vector representation to create a combined input representation for the autoencoder model.

## Step 3: Generating Target Skeletons from SMILES

To generate targets for the machine learning, we use RDKit to convert SMILES strings into molecular skeletons. RDKit provides functions to generate molecular skeletons from SMILES representations. These skeletons capture the core structural features of the molecules.

## Step 4: Rescaling Generated Skeletons for Consistent Scaling

To ensure consistent scaling of the generated molecular skeletons, we perform the following steps:

- First, we rescale the generated skeletons to a common scale or proportional size. This step ensures that all skeletons have a similar size and structural proportions, regardless of the original molecule's size.

- Next, we choose a target size for the molecular skeleton images. This size determines the dimensions of the final images that will be generated. It is important to consider an appropriate size that retains sufficient details and fits the majority (e.g., 99%) of the images.

- Finally, we scale all the molecular skeleton images to the chosen target size while maintaining the original structural proportions. This ensures that all images have a consistent scale and are ready for further processing or training in the autoencoder model.

By following these preprocessing steps, we can generate image representations of molecular skeletons that are suitable for training the autoencoder model for image generation tasks.

All of these steps are completed automatically by the program to allow for a full pipeline with minimal human interaction necessary.

