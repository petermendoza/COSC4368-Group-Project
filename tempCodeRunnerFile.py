class Q_Table:
    def __init__(self):
        self.q_table = np.zeros((grid_size, grid_size, 2, MAX_BLOCKS_PICK_UP + 1, MAX_BLOCKS_DROP_OFF + 1))

    def select_action(self, state, policy):
        if policy == 'random':
            return np.random.choice(num_actions)
        elif policy == 'greedy':
            return np.argmax(self.q_table[state])
        elif policy == 'exploit':
            if np.random.uniform(0, 1) < EPSILON:
                print("Random")
                return np.random.choice(num_actions)
            else:
                print("Exploit")
                return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, next_action, algorithm):
        print(next_state)
        print(self.q_table.shape)
        if algorithm == 'q-learning':
            self.q_table[state][action] += ALPHA * (
                        reward + GAMMA * np.max(self.q_table[next_state]) - self.q_table[state][action])
        elif algorithm == 'sarsa':
            self.q_table[state][action] += ALPHA * (
                        reward + GAMMA * self.q_table[next_state][next_action] - self.q_table[state][action])
