
This is a research project for a course in Continous Dynamical Systems. It was collaborative with Joseph Diaz. We decided to look at and learn about the DLDMD, Deep Learning Dynamic Mode Decomposition. [Jay Lago, https://arxiv.org/abs/2108.04433 ]

Since Jay Lago's code was open-source, we made slight modifications to see what contributions we can make to this existing code base. 

Our research paper of our findings can be found here: https://drive.google.com/file/d/1Pltk_F3AyVlBES0_kiUMldJl0pATp0i7/view?usp=sharing        
along with presentation slides:  
https://drive.google.com/file/d/1miaAqFOHJ4PpuPHIy08LFqo0sUagTKBR/view?usp=sharing

Here is a summary of changes we made:
1. Changed the NN Layers in the Autoencoder/Decoder to be Convolutional layers instead of Dense Layers [ie . DLDMD_TRY_CNN.py]
2. Since the above model was changed, needed to make small changes to references in the examples used + graphing functions [ie. Train_Duffing.py, Train_Rossler_With_CNN.py, .  ]
