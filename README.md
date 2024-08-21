# PAIReD jet tagging framework
A repository combining the full workflow of PAIReD tagger development.

## Introduction
[PAIReD jets](https://arxiv.org/abs/2311.11011) are new types of jets for improved reconstruction of hadronic decays of heavy objects such as the Higgs boson. This repository deals with the creation of a PAIReD tagger to classify these newly defined jets. From the production of PAIReD data sets to the training of a neural network and its testing, everything will be covered in this repository.

## Getting started: Recipe for the PAIReD training workflow
The individual steps of the workflow required to obtain a PAIReD tagger are described here in several chapters:
1. [Production of data sets](doc/workflow-recipe/production-datasets.md)
2. [Training a PAIReD tagger](doc/workflow-recipe/training-tagger.md)
3. [Evaluation of tagger performance]()

### Installing the framework
For installing this framework, go to the directory on your machine where you want to install it. Make sure that it is a reasonable place. Then, clone this repository to your current directory by running:
```bash
git clone https://github.com/JanGerritSchulz/PAIReD_jet_tagging.git
```

## Information on the produced PAIReD jet files
This framework enables the production of PAIReD jet files. Those are ROOT files containing a list of PAIReD jets, meaning each entry is one PAIReD jet. A summary on the stored variables and MC information can be found in the [PAIReD file content overview](./PAIReD_Data_Production/PFNano_to_PAIReD/notes/PAIReD_ROOT_file_content.md). This includes the definition of the labels and some selections on the seed AK4 jets used for building the PAIReD jets.

## Pre-trained PAIReD taggers
You can find a number of pre-trained PAIReD taggers in [PAIReD_Tagger_Training/trained-models](./PAIReD_Tagger_Training/trained-models/).