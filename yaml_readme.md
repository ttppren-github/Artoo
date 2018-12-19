# YAML configure file 
This file is used by main.py, to configure run in which environment and using which algorithm

## Three Major part
- train
- game
- agent
- coach

## train
Indicate if it is training mode.

    Value: True or False.
## game
- name  
    Which environment to use.  
    Type: string.
- games  
    How many games are there in this game?  
    Type: int
- render  
    If render the view.  
    Type: boolean.
    
## agent
- name 
- model_path

## coach
- name
- e_max: 100
- interval: 1
- learning_rate: 0.01
- reward_decay: 0.9
- e_greedy: 0.9
- save_model_feq: 10
