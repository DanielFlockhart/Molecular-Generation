# Molecular Generation With Variational Auto Encoding and Generative Adversarial Neworks
Developer Notes : This Project Is a work in progress, some functionality is WIP at the moment.

## Features

The project consists of the following main components :
1. **Preprocessing** - A [dataset](https://www.nature.com/articles/s41597-022-01142-7) of smiles is sorted and processed by accessing the dataset of smiles, rescaling and normalising the molecules so trained models has less representations and relationships to learn. This will improve the learning rate. Current Preprocessing

2. **Training** - Training is currently done using either a Vartiational Auto Encoder or Generative Adversaral Network.

3. **New Molecules Generation** - New molecules are generated.

A more detailed documentation of features will be updated soon.
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DanielFlockhart/Molecule-Generation.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Molecule-Generation
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage
4. In the case you want to use your own dataset please upload your txt of chemical names in this form. If you only wish to test the software, please skip to step 
 
   drugs.txt
   ```
   ["name1","name2","name3"...]
   ```


## Contribution

Contributions to this project are welcome! If you have any suggestions, improvements, or new features to propose, please submit a pull request. You can also report any issues or bugs by opening an issue on the project's GitHub repository.

When contributing, please follow the existing code style, write clear and concise commit messages, and provide appropriate documentation.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

## Credits

The project acknowledges the following resources for their contributions:

- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) - Data source for drug SMILES and names
- [RDKit](https://www.rdkit.org/) - Converting SMILEs to 2D Structures
- [LiverpoolUniversity](https://www.nature.com/articles/s41597-022-01142-7) - Dataset for theoretical predictions of new applications for existing compounds

Thank you for using the Molecular Generation project! We hope it proves to be useful for your chemical analysis and research.

## Current Work
- Write More Documentation and Explanations of VAE/GANs and the training Process
- Get VAE and GAN operational
- Make Flow Chart of program pipeline
- Decide whether inputs to networks should be binary (black or white) or have a continuous representations.
- Add possible Rotation and Transformation optimisations
- Add code efficiency optimisations
- Program possible otherways of representing molecules - E.g As a graph/matrix
- Add validity checks in for produced molecules
- Deploy program for easy use and access

## Future Project Ideas
- Convert Generated Images to back to SMILEs representation. Using Machine Learning?
- Docking/Simulation - Binding Affinity?