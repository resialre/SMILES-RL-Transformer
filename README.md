# SMILES-RL-Transformer
Experimentation with transformer-based actor models and transfer learning 
Transfer learning using teacher-student learning with tunable hybrid loss function (soft target and label loss). 

Transformer model performed as well, if not better, than RNN model at lower actor learning rates. Performance may improve with greater batch size and training steps. 

**Must be run within original SMILES-RL env***
https://github.com/MolecularAI/SMILES-RL/tree/main

The knowledge transfer process is defined in the mrun.py file. It requires the instantiation of a ModelDistillation class in the distill.py file. 

The initial model must be run with the prior file located in the original SMILES-RL page at all times. 
