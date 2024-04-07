import numpy as np

#defining the environment
GRID_SIZE = 5
NUM_ACTIONS = 6 #n, s, e, w, p, d
pick_up_cell = (1,1) 
drop_off_cell = (4,4)
MAX_BLOCKS_PICK_UP = 5
MAX_BLOCKS_DROP_OFF = 5
episodes = 10

# State space size for 3 agents
#state_space_size = (5, 5, 5, 5, 5, 5, 2, 2, 2, 6, 6, 6, 6, 6, 6)

# State space for 1 agent
# the first two index are 6 because the points rn go from 0-5 but need sto go from 1-5?
state_space_size = (6, 6, 2, 6, 6)

# set hyperparameters 
ALPHA = 0.1 #learning rate
GAMMA = 0.9 #discount factor
EPSILON = 0.1 #exploration vs exploitation factor

ACTIONS = {
    0: 'n', 
    1: 's', 
    2: 'e', 
    3: 'w', 
    4: 'p',
    5: 'd'
}


#Q-table class
class Q_Table:
    def __init__(self):
        self.q_table = np.zeros((state_space_size) + (NUM_ACTIONS,))
        print("Shape:", self.q_table.shape)

    def select_action(self, state, policy):
        # RANDOM policy: WORKING
        if policy == 'random':
            return np.random.choice(NUM_ACTIONS)
        
        # GREEDY policy: NOT TESTED
        elif policy == 'greedy':
            return np.argmax(self.q_table[state])
        
        # EXPLOIT policy: NEED FIXING (LOGIC INCORRECT)
        elif policy == 'exploit':
            if np.random.uniform(0, 1) < EPSILON:
                print("Random")
                return np.random.choice(NUM_ACTIONS)
            else:
                print("Exploit")
                return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action, algorithm):
        if algorithm == 'q-learning':
            self.q_table[state][action] += ALPHA * (
                        reward + GAMMA * np.max(self.q_table[next_state]) - self.q_table[state][action])
        elif algorithm == 'sarsa':
            self.q_table[state][action] += ALPHA * (
                        reward + GAMMA * self.q_table[next_state][next_action] - self.q_table[state][action])
        
        print("Updating: ")
        print("Old State: ", state)
        print("Action: ", ACTIONS[action])
        print("Qvalue: ", self.q_table[state][action])
        print("New State: ", next_state)
        print()


def aplop(i, j, x, a, d):
    applicable_operators = set()

    # Check if agent can move in each direction
    if i >= 1 and i < GRID_SIZE: #move right (east)
        applicable_operators.add(2)
        
    if i <= GRID_SIZE and i > 1: #move left (west)
        applicable_operators.add(3)
        
    if j <= GRID_SIZE and j > 1: #move down (south)
        applicable_operators.add(1)
        
    if j < GRID_SIZE and j >= 1: #move up (north)
        applicable_operators.add(0)

    # Check if agent can pick up or drop off a block
    if (i, j) == pick_up_cell and a > 0 and x == 0:
        applicable_operators.add(4)
    if (i, j) == drop_off_cell and x == 1 and d < MAX_BLOCKS_DROP_OFF:
        applicable_operators.add(5)

    return applicable_operators


def apply(i, j, x, a, d, action):
    new_i, new_j = i, j #initalizes new_i and new_j with curr position of robot
    new_x = x #initializes new_x to number if is robot holding block?
    new_a = a # initializes the new_a to number of blocks in pickup cell
    new_d = d #initializes the new_d to number of blocks in dropoff cell

    # Action == 0 means AGENT moves UP / NORTH
    if action == 0:
        new_j += 1 
        
    # Action == 1 means AGENT moves DOWN / SOUTH
    elif action == 1:
        new_j -= 1 
    
    # Action == 2 means AGENT moves RIGHT / EAST
    elif action == 2:
        new_i += 1 
        
    # Action == 3 means AGENT moves LEFT / WEST
    elif action == 3:
        new_i -= 1 
        
    # Action == 4 means AGENT PICKUPS BLOCK
    elif action == 4:
        new_a -= 1 #decrement i.e. taking block from pickup
        new_x = 1 #increment i.e. agent is now holding block
        
    # Action == 5 means AGENT DROPPOFF
    elif action == 5:
        new_d += 1 #increment i.e. agent put into dropoff
        new_x = 0 #decrement i.e. agent is no longer holding block

    return new_i, new_j, new_x, new_a, new_d #return the new state

def simulate_episode(robot, algorithm, policy):
    state = (1, 1, 0, MAX_BLOCKS_PICK_UP, 0)  # Initial state
    print("State: " , state)
    total_reward = 0

    while state[4] != MAX_BLOCKS_DROP_OFF:
        applicable_operators = aplop(*state)
        print("Applicable operators: ", applicable_operators)
        
        
        # If able to pick up it will pick up
        if 4 in applicable_operators:
                next_state = apply(*state,4)
                action = 4
                print("Action: ", action)
                
        # If able to drop off agent will drop off 
        elif 5 in applicable_operators:
                next_state = apply(*state,5)
                action = 5
                print("Action: ", action)
                
        # Other wise choose action based on policy
        else:
            action = robot.select_action(state, policy)
            print("Action: ", action)

            if action in applicable_operators:
                next_state = apply(*state, action)
                
        print("next state: ", next_state)
        reward = -1  # Default reward for moving
        if (state[0:2] == pick_up_cell and action == 'p') or (state[0:2] == drop_off_cell and action == 'd'):
            reward = 13  # Reward for picking up or dropping off
        robot.update_q_table(state, action, reward, next_state, robot.select_action(next_state, policy), algorithm)
        total_reward += reward
        state = next_state

    return total_reward


def main():
    _q_learning = Q_Table()
    _sarsa = Q_Table()

    for episode in range(episodes):
        reward_q_learning = simulate_episode(_q_learning, 'q-learning', 'random')
        reward_sarsa = simulate_episode(_sarsa, 'sarsa', 'random')

        if episode % 100 == 0:
            print(f"Episode {episode}: Q-learning reward = {reward_q_learning}, SARSA reward = {reward_sarsa}")

    print("Q-learning Q-table:")
    print(_q_learning.q_table)
    print("SARSA Q-table:")
    print(_sarsa.q_table)


if __name__ == "__main__":
    main()