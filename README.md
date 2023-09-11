# Molecular Generation With Variational Auto Encoding and Generative Adversarial Neworks
Developer Notes :
- This Project Is a work in progress
- Data has been preprocessed for use with machine learning. It is not included in this repo as it is 500mb+

## Features
![Project Pipeline](image.png)
The project consists of the following main components :
1. **Preprocessing** - A [dataset](https://www.nature.com/articles/s41597-022-01142-7) of smiles is sorted and processed by accessing the dataset of smiles, rescaling and normalising the molecules so trained models has less representations and relationships to learn. This will improve the accuracy.


----- Work In Progress -----

2. **Training** - Training is currently done using either a Vartiational Auto Encoder or Generative Adversaral Network.

3. **New Molecules Generation** - New molecules are generated.

A more detailed documentation of features will be updated soon.
## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/DanielFlockhart/Molecular-Generation.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Molecular-Generation
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Code is not yet set up with User Interface for usage - Will update README.md at a later stage

2. To Preprocess data
   ```bash
      cd src
      python main.py

      ------------------------- Initialising -------------------------
      ------------------------- Preprocessing Data -------------------------
      ------------------------- Downloading Images -------------------------
      100%|████████████████████████████████████| 48167/48167 [06:09<00:00, 130.33it/s]
      ------------------------- Scaling Images -------------------------
      100%|████████████████████████████████████| 47471/47471 [04:02<00:00, 195.92it/s]
      ------------------------- Preprocessing Summary -------------------------
      Generated From Smiles        : 98.55502730084913%
      Bound Used                   : 617.0
      Amount of Images Above Bound : 902/47471

      ```



## Contribution

Contributions to this project are welcome! If you have any suggestions, improvements, or new features to propose, please submit a pull request. You can also report any issues or bugs by opening an issue on the project's GitHub repository.

When contributing, please follow the existing code style, write clear and concise commit messages, and provide appropriate documentation.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute it as per the terms of the license.

## Credits

The project acknowledges the following resources for their contributions:

- [RDKit](https://www.rdkit.org/) - Converting SMILEs to 2D Structures
- [LiverpoolUniversity](https://www.nature.com/articles/s41597-022-01142-7) - Dataset for theoretical predictions of new applications for existing compounds

Thank you for using the Molecular Generation project! We hope it proves to be useful for your chemical analysis and research.


## To Do
- Reverse image to smiles - Molecular Profiles
- Train a full Model
- Write Paper 2
- Under the other conditions version

