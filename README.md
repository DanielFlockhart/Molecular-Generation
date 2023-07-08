# Molecular Generation With Variational Auto Encoding and Generative Adversarial Neworks
Developer Notes : This Project Is a work in progress, some functionality is WIP at the moment.

## Features

The project consists of the following main components:

1. **Webscraping**: The project includes a webscraping module that fetches drug SMILES and names from reliable sources. This data will serve as the basis for chemical compound clustering.

2. **Clustering**: The clustering module utilizes agglomerative clustering with levenshtein distance to cluster the chemical compounds based on their SMILES. It computes the similarity between compounds and assigns them to appropriate clusters.

3. **Chemical Identification**: This module takes a SMILE and outputs the predicted chemical.

4. **SMILE To Structure**: This module takes a SMILE and outputs the predicted chemical.

5. **Preprocessing**

6. **Training**

7. **New Molecules Generation**


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

## To Do

1. Admin/Documentation
   - Finish Readme and write up proper documentation

2. Preprocessing
   - Colour
   - Rotation and Transformation optimisation
   - Simplification
   - Choose new Arbritrary Standard Deviation
   - Do preprocessing in stages so I save time

3. ML
   - Get VAE working
   - Get GAN working

4. Postprocessing
   - Image to Smile


5. Deployment
   - Get program generating new Images

6. Evaluation
   - Evaluate if it is valid molecule
   - Docking?


7. General Programming
   - Comment Code
   - Remove any try/excepts and work out why errors are there
      - Doesn't generate smiles with incorrect amount of bonds to an atom
      - Doesn't generate smiles with smile name that can't be a file type
   - Reshuffle File System
   - Remove Debugging Messages
   - Fix Valence Issue
   - Make Flow Chart
