# COSC4368-Group-Project

### Group Members

#### Bryant Huynh, Peter Mendoza, Maryann Tran, Ryan Park

### About the Project
We were tasked to create a Grid World with the goal of having three agents, red, blue and black, learn the most promising paths to pickup a total of 5 blocks from any of the 3 pickup locations and dropoff to any of the 3 dropoff locations. 

### Prerequisities 

* NumPy
```sh
pip install numpy
```

* Matplotlib
```sh
python -m pip install -U matplotlib
```
### Seed 
In the main function, line 479

```sh
    random.seed(CHANGE TO PREFERRED NUMBER HERE)
```
The seed can be changed to any integer of your choice, and is used solely to prevent replicating results.


### How to Change Policy
The three agents can learn through any of three policies: Random, Greedy, and Exploit. 

In the main function, line 509
```sh
 simulate_episodes(num_steps, env, q_table, alpha, gamma, epsilon, 'CHANGE POLICY HERE', 'CHANGE ALGORITHM HERE', experimentNum)
```
You can change the policy by typing the correct policy (syntax matters) : 'random', 'greedy', or 'exploit

### How to Change Algorithm
The algorithm can be changed to either of the two algorithms: Q-Learning & SARSA

In the main funciton, line 509
```sh
 simulate_episodes(num_steps, env, q_table, alpha, gamma, epsilon, 'CHANGE POLICY HERE', 'CHANGE ALGORITHM HERE', experimentNum)
```
You can change the algorithm by typing the correct policy (syntax matters) : 'q-learning' or 'sarsa'

### How to Change Learning rate
The learning rate can be changed in the main function , line 499

```sh
    alpha = CHANGE TO PREFERRED NUMBER
```