# Input Synthesis
The goal of this module is to allow for easier use of the model, allowing users to not have to worry about providing 
the model explicitly with starting molecules and goal conditions

This module will use some type of NLP to process a user input like

```Generate me a molecule starting from benzene that can be used to treat schizophrenia```

```Generate me some possible molecules for treatment of depression```

To create a starting molecule and conditions to pass into the model.

This will allow for the user to save time having to find a possible starting Molecule, finding its smile, passing it through the ChemBERTa Model (Although this is implemented in the preprocessing of the dataset auotmatically anyway), getting the specific conditions required and then passing it through the model.

In Summary, this module will make the program easier for average people to use for drug discovery.

This may be complex to implement.

Possible methods :
- Transformer
- Some sort of webscraping combined with self attention
- Search Through Papers for possible treatments

Model should be able to generate multiple potential molecules from multiple different starting molecules


## Take dataset of illness's and their corresponding treatments do some sort of processing on them and generate multiple possible starting molecules and conditions

## Users should be able to manually enter starting molecule

