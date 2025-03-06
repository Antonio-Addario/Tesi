# Generating Pull Request Messages with LLMs

This repository contains the code developed for my Bachelor's thesis in Computer Science at the University of L'Aquila. The thesis explores the use of Large Language Models (LLMs) to automatically generate messages for pull requests.

## Abstract

Pull request messages provide essential context for understanding code modifications in software development. This project investigates the use of LLMs to automate the generation of these messages, aiming to improve the efficiency and quality of the software development process.

## Project Description

The project utilizes the LLAMA LLM to generate pull request titles and body messages based on metadata and code changes. The approach involves:

* Preprocessing pull request data.
* Retrieving contextual information.
* Integrating the LLM for message generation.
* Postprocessing the generated messages.
* Saving the results.

## Dependencies

The project relies on the following dependencies:

* Python 3.x
* LLAMA LLM
* Pymongo
* Scikit-learn
* NLTK
* SentenceTransformer
* Other libraries specified in `requirements.txt`

## Evaluation

The quality of the generated pull request messages is evaluated using the following metrics:

* BLEU (Bilingual Evaluation Understudy) 
* METEOR (Metric for Evaluation of Translation with Explicit ORdering) 
* Cosine Similarity


## Acknowledgements

Thanks to Prof. Juri Di Rocco and Dr. Claudio Di Sipio for their guidance and support throughout this project.
