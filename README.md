# Domain_transfer_network
This repository is made in connection with my Master thesis regarding domain transfering, with paper found here "link". It includes the code for a network (give name) which has learned to transfer a person from a random 'in-the-wild' enviroment, to another, namely to a proffesional studio enviroment.


## How to use project 

Training data used for this project can be found from "link" (in-studio/img/hish-resolution, customer/img/high_resolution). Due to XXX reasons I can't share the dataset, but after downloading both dataset, run **aasdasfa.py** to generate dataset we used. This dataset will need to be added to 'data/' folder (if you want to train the network). To use a pretrained model, download model here "link".


There exists 4 executable python files, all which were used using : pythonx.y.z, torchx.y.z, cuda, for simplicity use conda enviroment "".

**use_model.py**: loads model, takes all images (resizes to 750x1106) from data/use_model_input, and returns output to data/use_model_output

**training.py**: uses data from  data/train.py and returns a model in data/model.

**validation.py**: uses model calculates XXX score, and prints the score.

**last_file.py**: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed"


## paper results

comparison of scores, key-points






