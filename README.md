# Reinforcement Learning for 3-Player Chinese Checkers

This project enables the training of an agent to play a version of the 3-player game Chinese Checkers.

A further descripton of the project and a more detailed presentation of the results can be found in my Bachelor's thesis"Reinforcement Learning for 3-player Chinese Checkers". You can read it [here](https://github.com/davidschulte/alpha-thesis/blob/master/Thesis.pdf).

### Results
The trained agent showed clearly improved over the training iterations. One evaluation measure was the competition against a simply designed greedy agent (The assigned scores correspend to the placement in the game). 
![Alt text](mcts_vs_greedy.png?raw=true "Evaluation against Greedy Agent")

### Trained models
Because of their large size, this project does not contain the trained models used in the thesis.

### Contributors and Credits
I used an existing repository by Surag Nair, which implemented the logic of AlphaZero by DeepMind and applied it to several 2-player games.
Please have a look at it, as it greatly helped me realizing the concept of AlphaZero and deepened my understanding of Reinforcement Learning.
https://github.com/suragnair/alpha-zero-general/

The overall structure and the code were changed to enable training in a 3-player game.
The training process was modified, such that several games are played simultaneously, making better use of the pipeline in Tensorflow and leading to a significant speed-up.
The game Chinese Checkers was implement with a graphical interface.

