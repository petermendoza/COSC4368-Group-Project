import numpy as np

#defining the environment
grid_size = 5
num_actions = 6 #n, s, e, w, p, d
pick_up_cell = (1,1) 
drop_off_cell = (4,4)
MAX_BLOCKS_PICK_UP = 5
MAX_BLOCKS_DROP_OFF = 5
episodes = 1000

# set hyperparameters 
ALPHA = 0.1 #learning rate
GAMMA = 0.9 #discount factor
EPSILON = 0.1 #exploration vs exploitation factor

ACTIONS = {
    'n': 0,
    's': 1,
    'e': 2,
    'w': 3,
    'p': 4,
    'd': 5
}


#Q-table class
class Q_Table:
    def __init__(self):
        self.q_table = np.zeros((grid_size, grid_size, 2, MAX_BLOCKS_PICK_UP + 1, MAX_BLOCKS_DROP_OFF + 1, num_actions))

    def select_action(self, state, policy):
        if policy == 'random':
            return np.random.choice(num_actions)
        elif policy == 'greedy':
            return np.argmax(self.q_table[state])
        elif policy == 'exploit':
            if np.random.uniform(0, 1) < EPSILON:
                return np.random.choice(num_actions)
            else:
                return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action, algorithm):
        if algorithm == 'q-learning':
            self.q_table[state][action] += ALPHA * (
                        reward + GAMMA * np.max(self.q_table[next_state]) - self.q_table[state][action])
        elif algorithm == 'sarsa':
            self.q_table[state][action] += ALPHA * (
                        reward + GAMMA * self.q_table[next_state][next_action] - self.q_table[state][action])


def aplop(i, j, x, a, d):
    applicable_operators = set()

    # Check if agent can move in each direction
    if i > 0:
        applicable_operators.add('n')
    if i < grid_size - 1:
        applicable_operators.add('s')
    if j > 0:
        applicable_operators.add('w')
    if j < grid_size - 1:
        applicable_operators.add('e')

    # Check if agent can pick up or drop off a block
    if (i, j) == pick_up_cell and a > 0:
        applicable_operators.add('p')
    if (i, j) == drop_off_cell and x == 1 and d < MAX_BLOCKS_DROP_OFF:
        applicable_operators.add('d')

    return applicable_operators


def apply(i, j, x, a, d, action):
    new_i, new_j = i, j #initalizes new_i and new_j with curr position of robot
    new_x = x #initializes new_x to number if is robot holding block?
    new_a = a # initializes the new_a to number of blocks in pickup cell
    new_d = d #initializes the new_d to number of blocks in dropoff cell

    if action == 'n':
        new_i -= 1 #decrement row index indicating robot step north
    elif action == 's':
        new_i += 1 #increment rwo index indicating robot moves one step south
    elif action == 'w':
        new_j -= 1 #decrement row index indicating robot step west
    elif action == 'e':
        new_j += 1 #increment indicating robot east
    elif action == 'p':
        new_a -= 1 #decrement i.e. taking block from pickup
        new_x = 1 #increment i.e. agent is now holding block
    elif action == 'd':
        new_d += 1 #increment i.e. agent put into dropoff
        new_x = 0 #decrement i.e. agent is no longer holding block

    return new_i, new_j, new_x, new_a, new_d #return the new state

def simulate_episode(robot, algorithm, policy):
    state = (0, 0, 0, MAX_BLOCKS_PICK_UP, 0)  # Initial state
    total_reward = 0

    while state[0:2] != drop_off_cell or state[4] < MAX_BLOCKS_DROP_OFF:
        applicable_operators = aplop(*state)
        action = robot.select_action(state, policy)
        print(action)
        if action in applicable_operators:
            next_state = apply(*state, action)
            reward = -1  # Default reward for moving
            if (state[0:2] == pick_up_cell and action == 'p') or (state[0:2] == drop_off_cell and action == 'd'):
                reward = 13  # Reward for picking up or dropping off
            robot.update_q_table(state, ACTIONS[action], reward, next_state, robot.select_action(next_state, policy), algorithm)
            total_reward += reward
            state = next_state

    return total_reward


def main():
    _q_learning = Q_Table()
    _sarsa = Q_Table()

    for episode in range(episodes):
        reward_q_learning = simulate_episode(_q_learning, 'q-learning', 'exploit')
        reward_sarsa = simulate_episode(_sarsa, 'sarsa', 'exploit')

        if episode % 100 == 0:
            print(f"Episode {episode}: Q-learning reward = {reward_q_learning}, SARSA reward = {reward_sarsa}")

    print("Q-learning Q-table:")
    print(_q_learning.q_table)
    print("SARSA Q-table:")
    print(_sarsa.q_table)


if __name__ == "__main__":
    main()