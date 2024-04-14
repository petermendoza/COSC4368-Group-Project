# COSC4368-Group-Project

### Group Members

#### Bryant Huynh, Peter Mendoza, Maryann Tran, Ryan Park

## How to run experiments

Since each experiment is just a change in the hyperparameters, seed, policy used, and choosing between Q-Learning and SARSA, we modularized the methods and you only need to change certain things in the main function to run each separate experiment.  
The first thing you can set in main is a seed, which can be changed to any integer of your choice, and is used solely for replicating results.  
You can change alpha (learning rate), gamma (discount factor), and epsilon (exploration vs exploitation factor), in lines 499-501, which are changed between experiments, so you can change those parameters to your needs.  
You then need to change experimentNum in line 504 to whatever experiment you are running, but it really only matters for experiment 4 as the pickup locations change.  
Then you can change the number of steps you will be using with num_steps.  
To actually run the agents, you will use the simulate_episodes function, with parameters (num_steps, environment, q_table, alpha, gamma, epsilon, policy, learning type, experimentNum). For the most part, you will just want to change the num_steps to either a flat 500 for the initial setup seen in most experiments, then you also want to change the policy and learning type. The options for policy are 'random', 'greedy', and 'exploit', and the options for learning are 'q-learning' and 'sarsa'.  

The rest is just for visualization and such, but that should be all you need for running every experiment.
