- Train Model
- Take Smiles and Generate Images 
- Preprocess Images - focal scaling
- Convert images to vector Representation
- Input to a network (GAN, VAE)
- Generate New Potential Analogues

- Working on finding a way of making VAE positional and scale invariant

# Molecular Generation With Variational Auto Encoding and Generative Adversarial Neworks
Developer Notes : This Project Is a work in progress, some functionality is WIP at the moment.

This Chemical Smiles Toolkit has a variety of features including clustering chemical compounds based on their SMILES (Simplified Molecular Input Line Entry System) representation and provides a user-friendly interface to input a SMILES string and obtain a cluster of similar chemicals along with their respective SMILES. In addition, the user can input a SMILE and recieve a 2D Structure in return.

There is no requirement to cluster the default data on first use, it has already been clustered using 100 clusters.

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


5. Launch the program:

   ```bash
   python main.py

   ```
6. Follow Instructions

   ```console
   ---------- Welcome chemical SMILES toolkit ----------

    The Github repository comes with a pre-clustered dataset of 1411 Psychoactive Substances with 100 clusters as an example.
    Feel free to use this dataset or cluster your own dataset.
    Please choose from the follow options to continue:

    1. Get similar SMILE to a given SMILE with current clusters
    2. Re-cluster data with a different number of clusters
    3. Re-cluster data with a different dataset
    4. Convert a SMILE to a 2D structure and display it
    5. Get the name of a chemical from a SMILE  
   ```

7. Getting Similar Chemicals
   ```console
   Enter a smile: CCC(CC1=CNC2=CC=CC=C21)N

   Alpha-methyltryptamine       CC(CC1=CNC2=CC=CC=C21)N
   Alpha-ethyltryptamine        CCC(CC1=CNC2=CC=CC=C21)N
   Alpha,N-DMT                  CC(CC1=CNC2=CC=CC=C21)NC
   5-MeO-AMT                    CC(CC1=CNC2=C1C=C(C=C2)OC)N
   Alpha,N,O-TMS                CC(CC1=CNC2=C1C=C(C=C2)OC)NC
   5-Fluoro-AMT                 CC(CC1=CNC2=C1C=C(C=C2)F)N
   6-fluoro-AMT                 CC(CC1=CNC2=C1C=CC(=C2)F)N
   MethylbenzodioxolylbutanamineCCCC(C)(C1C2=CC=CC=C2OO1)N
   Benzodioxolylbutanamine      CCCC(C1C2=CC=CC=C2OO1)N
   Naphthylaminopropane         CC(CC1=CC2=CC=CC=C2C=C1)N

   ```
8. Converting a SMILE to a 2D structure and display it.
   ```console
   Enter a smile: CCC(CC1=CNC2=CC=CC=C21)  
   ```
   Displayed Image (The file name of the image is the name of the Chemical)

   ![Alt Text](data/2D-Structures/3-butyl-1H-indole.png)

9. Getting the name of a chemical from a SMILE
    ```console
   Enter a smile: CCC(CC1=CNC2=CC=CC=C21)  
   The SMILE corresponds to the chemical -> 3-butyl-1H-indole

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
- [LiverpoolUniversity](https://www.nature.com/articles/s41597-022-01142-7) - Organic materials repurposing, a data set for theoretical predictions of new applications for existing compounds


Thank you for using the Molecular Generation project! We hope it proves to be useful for your chemical analysis and research.


Issues - Colour is still taken into account in images and so is scaling and position