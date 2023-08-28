# Training Steps for Molecular Skeleton Image Generation Model

This markdown file describes the training steps for the molecular skeleton image generation model using the preprocessed SMILES and condition vectors.

## Step 1: Input Preparation

- Take the preprocessed SMILES vectors and condition vectors, which have been previously concatenated.
- Pass the concatenated vector representation through the encoder network. The encoder network encodes the input vectors into a lower-dimensional latent representation.

## Step 2: Latent Vector Concatenation

- Concatenate the latent dimension vector obtained from the encoder with the condition vector.
- This concatenated vector serves as the input for the decoder network, allowing the decoder to generate a reconstructed representation of the input.

## Step 3: Reconstruction and Target Vector

- Pass the concatenated vector through the decoder network, which decodes the input to generate a reconstructed representation.
- The target vector for training is the ground truth molecule images that have been previously converted into vectors during the preprocessing step using RDKit's manual creation.

## Step 4: Loss Calculation and Network Update

- Calculate the loss between the reconstructed representation from the decoder and the target vector.
- Use an appropriate loss function, such as mean squared error (MSE), to measure the discrepancy between the reconstructed representation and the target vector.
- Update the parameters of the encoder and decoder networks based on the calculated loss using backpropagation and gradient descent.
- Repeat the training process by iterating through the dataset multiple times (epochs) to optimize the network's performance.

## Step 5: Training the Model

- Train the model by iteratively performing the above steps
