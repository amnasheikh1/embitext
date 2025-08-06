# EmbiText: Embracing Ambiguity by Annotation, Recognition and Generation of Pronominal Reference with Event-Entity Ambiguity

This study introduces a new dataset \textbf{\scalebox{1.2}{E}MBITEX\scalebox{1.2}{T}} to model ambiguity in language by navigating through ambiguity surrounding pronominal reference to entity or event. 



 ## Objective 
This research aims to explore the following research directions: 
- How can the ambiguity inherent in pronominal references between entities and event be identified, annotated, and modeled in Natural Language?
- Are LLMs capable of embracing ambiguity in Natural language rather than resolving it?


 ## Data 

Two Datasets are used for this study, 
- The Real-world dataset comes from a paper by [Zeldes et al]
([https://doi.org/10.48550/arXiv.1711.00350](https://aclanthology.org/2025.cl-1.3/)) and is publicly available. (https://github.com/amir-zeldes/gum.git) folder.

- Data examples generated from LLMs. 

The datasets are compiled together are provided in [data] folder. Dataset is split into train set [data/train_data.csv](data/train_data.csv) and test set [data/test_data](data/test_data.csv).


## Code 
- Training loop that fine-tunes RoBERTa for Pronoun Probability Inference to predict entity reference probabilities, resulting in event eference probabilities as their complement in [code/training_loop_entity.py](code/training_loop_entity.py)
- Training loop that fine-tunes RoBERTa forf Pronoun Probability Inference to predict event reference probabilities, resulting in event eference probabilities as their complement in [code/training_loop_event.py](code/training_loop_event.py)
- Code for promting LLMs to generate ambiguous text examples in [code/text_generation_experiment.ipynb](code/text_generation_experiment.ipynb)


Note: This Paper is submitted as a dual submission to both CODI and CRAC, per the CODI‑CRAC 2025 joint Call For Papers. 
