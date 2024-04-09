import numpy as np
import random
import matplotlib.pyplot as plt


class RLEnvironment:
    def __init__(self, grid_size=5, num_agents=3):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.grid = np.zeros((grid_size, grid_size),
                             dtype=int)  # Grid initialization
        self.agent_colors = ['red', 'black', 'blue']  # Colors for the agents

        # Initialize pickup and dropoff locations
        self.pickup_locations = [(0, 4), (1, 3), (4, 1)]
        self.dropoff_locations = [(0, 0), (2, 0), (3, 4)]

        # Initialize agent locations and colors
        self.agent_info = [{'location': (3, 3), 'color': 'red', 'carrying': False},  # Agent ID 0
                           {'location': (4, 3), 'color': 'blue', 'carrying': False},  # Agent ID 1
                           {'location': (1, 3), 'color': 'black', 'carrying': False}]  # Agent ID 2

        # Initialize block counts at pickup and dropoff locations
        self.pickup_blocks = [5] * num_agents
        self.dropoff_blocks = [0] * num_agents

    def aplop(self, x, y):
        applicable_operators = set()

        # Check if agent can move in each direction
        if x > 0:  # move left
            target_location = x-1
            location_vacant = all(
                agent['location'] != target_location for agent in self.agent_info)
            if location_vacant:
                applicable_operators.add(0)
                
        if x < self.grid_size-1:  # move right
            target_location = x+1
            location_vacant = all(
                agent['location'] != target_location for agent in self.agent_info)
            if location_vacant:
                applicable_operators.add(1)
                
        if y > 0:  # move down
            target_location = y-1
            location_vacant = all(
                agent['location'] != target_location for agent in self.agent_info)
            if location_vacant:
                applicable_operators.add(2)
                
        if y < self.grid_size-1:  # move up
            target_location = y+1
            location_vacant = all(
                agent['location'] != target_location for agent in self.agent_info)
            if location_vacant:
                applicable_operators.add(3)

        return applicable_operators

    def move_agent(self, agent_id, action):

        x, y = self.agent_info[agent_id]['location']

        aplop = self.aplop(x, y)
        
        if not aplop:  # edge case where aplop is empty
            return 'none'  # stay still, agent is trapped
        
        if action not in aplop:  # action chosen not valid
            action = random.choice(list(aplop))  # choose random action
        
        # Move Left
        if action == 0:
            x -= 1
        
        # Move Right    
        elif action == 1:
            x += 1
            
        # Move Down
        elif action == 2:
            y -= 1
            
        # Move Up
        elif action == 3:
            y += 1

        # Update agent location
        self.agent_info[agent_id]['location'] = (x, y)

        return action
    
    def select_action(self, policy, q_table, epsilon):
        
        # RANDOM Policy
        if (policy == 'random'):
            return random.randint(0,3)
        
        # GREEDY Policy
        elif (policy == 'greedy'):
            print('GREEDY')
            
        # EXPLOIT Policy
        elif (policy == 'exploit'):
            print('EXPLOIT')

    def plot_world(self):
        """
        Plot the current state of the world.
        """
        fig, ax = plt.subplots()

        # Plot grid
        ax.grid(True)
        # Keep the x-axis ticks as they are
        ax.set_xticks(np.arange(0, self.grid_size, 1))
        # Keep the y-axis ticks as they are
        ax.set_yticks(np.arange(0, self.grid_size, 1))

        # Plot pickup locations
        for loc in self.pickup_locations:
            # Brown square, switch x and y coordinates
            ax.plot(loc[1], loc[0], color='brown', marker='s', markersize=10)

        # Plot dropoff locations
        for loc in self.dropoff_locations:
            # Green square, switch x and y coordinates
            ax.plot(loc[1], loc[0], color='green', marker='s', markersize=10)

        # Plot agents
        for agent in self.agent_info:
            color = agent['color']
            loc = agent['location']
            # Circle, switch x and y coordinates
            ax.plot(loc[1], loc[0], marker=6, color=color, markersize=10)

        ax.set_title('World State')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Flip the y-axis
        plt.show()


# PRANDOM
def PRANDOM(steps, env, q_table, alpha, gamma, epsilon, policy, learning):
    episode_count = 0
    
    for step_count in range(steps):
        for i in env.agent_info:
            
            # Selects action based on policy choosen
            action = env.select_action(policy, q_table, epsilon)
            
            location = i['location']
            (x, y) = location

            agent_color = i['color']
            match agent_color:
                case 'red':
                    agent_id = 0
                case 'blue':
                    agent_id = 1
                case 'black':
                    agent_id = 2

            # If agent is in a pickup location and is not carrying a block
            if (x, y) in env.pickup_locations and i['carrying'] == False:
                pickup_index = env.pickup_locations.index((x, y))
                
                # If there is a block at the location: PICK UP
                if env.pickup_blocks[pickup_index] > 0:
                    env.pickup_blocks[pickup_index] -= 1
                    
                    i['carrying'] = True
                    action = 8
                    reward = 15
                    max_q_value = np.max(q_table[4:7, y, x])
                
                # If there is not a block at the location: MOVE
                else:  
                    response = env.move_agent(agent_id, action)

            # If agent is at a drop off location and is carrying a block
            elif (x, y) in env.dropoff_locations and i['carrying'] == True:
                dropoff_index = env.dropoff_locations.index((x, y))
                
                # If there is space to drop off a block then: DROP OFF
                if env.dropoff_blocks[dropoff_index] < 5:
                    env.dropoff_blocks[dropoff_index] += 1
                    i['carrying'] = False
                    
                    action = 9
                    reward = 15
                    max_q_value = np.max(q_table[0:3, y, x])
                
                # If there is no space to drop off a block then: MOVE
                else: 
                    # +4 because agent is currently carrying a block
                    response = env.move_agent(agent_id, action) + 4
                    
            else:
                
                if i['carrying'] == False:
                    response = env.move_agent(agent_id, action)
                else:
                    response = env.move_agent(agent_id, action) + 4
                    
            if response == 'none':
                # agent is trapped in a corner, will not update q-values
                continue
            
            new_location = i['location']
            (new_x, new_y) = new_location
            
            if (action == 8 or action == 9):
                # Q Learning for drop off and pick up
                q_table[action][y][x] = (1-alpha)*q_table[action][y][x] + alpha*(reward+gamma*max_q_value)
                
                # --- TODO: ADD SARSA
                continue
            
            if i['carrying'] == False:
                max_q_value = np.max(q_table[0:3, new_y, new_x])
            else:
                max_q_value = np.max(q_table[4:7, new_y, new_x])

            # Checks to see if pickup/dropoff in next state is an applicable action for calculating q_max
            if i['carrying'] == False and (new_x, new_y) in env.pickup_locations:
                pickup_index = env.pickup_locations.index((new_x, new_y))
                if env.pickup_blocks[pickup_index] > 0:
                    max_q_value = max(max_q_value, q_table[8, new_y, new_x])
                    
            if i['carrying'] == True and (new_x, new_y) in env.dropoff_locations:
                dropoff_index = env.dropoff_locations.index((new_x, new_y))
                if env.dropoff_blocks[dropoff_index] < 5:
                    max_q_value = max(max_q_value, q_table[9, new_y, new_x])

            q_table[response][y][x] = (1-alpha)*q_table[response][y][x] + alpha*(-1+gamma*max_q_value)

            # ADD SARSA

        if all(blocks == 5 for blocks in env.dropoff_blocks) and all(blocks == 0 for blocks in env.pickup_blocks):
            # Resetting pickup and dropoff
            for i in range(len(env.dropoff_blocks)):
                env.dropoff_blocks[i] = 0
            for i in range(len(env.pickup_blocks)):
                env.pickup_blocks[i] = 5
            episode_count += 1
            
    return episode_count


def PEXPLOIT(steps, env, q_table, alpha, gamma, epsilon, ):
    episode_count = 0
    
    for step_count in range(steps):
        for i in env.agent_info:
            
            action = random.randint(0, 3)
            
            location = i['location']
            (x, y) = location
            agent_color = i['color']
            match agent_color:
                case 'red':
                    agent_id = 0
                case 'blue':
                    agent_id = 1
                case 'black':
                    agent_id = 2

            # If agent is in a pickup location and is not carrying a block
            if (x, y) in env.pickup_locations and i['carrying'] == False:
                pickup_index = env.pickup_locations.index((x, y))
                
                # If there is a block in the space: PICK UP
                if env.pickup_blocks[pickup_index] > 0:
                    env.pickup_blocks[pickup_index] -= 1
                    i['carrying'] = True
                    action = 4
                    
                    # Q-LEARNING
                    q_table[action][y][x] = (1-alpha)*q_table[action][y][x] + alpha*(
                        15+gamma*np.max(q_table[0:3, y, x]))  # makes sure a' is applicable
                    
                    # ADD SARSA

                    continue
                
                # If there is no block in the space: MOVE
                else:
                    response = env.move_agent(agent_id, action)

            # 
            elif (x, y) in env.dropoff_locations and i['carrying'] == True:
                dropoff_index = env.dropoff_locations.index((x, y))
                
                # Drop off a block if there is space
                if env.dropoff_blocks[dropoff_index] < 5:
                    env.dropoff_blocks[dropoff_index] += 1
                    i['carrying'] = False
                    action = 5
                    
                    # Q-LEARNING
                    q_table[action][y][x] = (1-alpha)*q_table[action][y][x] + alpha*(
                        15+gamma*np.max(q_table[0:3, y, x]))  # makes sure a' is applicable
                    # ADD SARSA

                    continue
                
                else:  # if dropoff is full
                    response = env.move_agent(agent_id, action)
            else:
                response = env.move_agent(agent_id, action)
            if response == 'none':
                # agent is trapped in a corner, will not update q-values
                continue
            new_location = i['location']
            (new_x, new_y) = new_location

            # CHECK MAX Q VALUE FOR CARRYING TOO
            max_q_value = np.max(q_table[0:3, new_y, new_x])

            # Checks to see if pickup/dropoff in next state is an applicable action for calculating q_max
            if i['carrying'] == False and (new_x, new_y) in env.pickup_locations:
                pickup_index = env.pickup_locations.index((new_x, new_y))
                if env.pickup_blocks[pickup_index] > 0:
                    max_q_value = max(max_q_value, q_table[8, new_y, new_x])
            if i['carrying'] == True and (new_x, new_y) in env.dropoff_locations:
                dropoff_index = env.dropoff_locations.index((new_x, new_y))
                if env.dropoff_blocks[dropoff_index] < 5:
                    max_q_value = max(max_q_value, q_table[9, new_y, new_x])

            q_table[response][y][x] = (1-alpha)*q_table[response][y][x] + alpha*(-1+gamma*max_q_value)
            # ADD SARSA

        if all(blocks == 5 for blocks in env.dropoff_blocks) and all(blocks == 0 for blocks in env.pickup_blocks):
            # Resetting pickup and dropoff
            for i in range(len(env.dropoff_blocks)):
                env.dropoff_blocks[i] = 0
            for i in range(len(env.pickup_blocks)):
                env.pickup_blocks[i] = 5
            episode_count += 1
    return episode_count


def main():
    np.set_printoptions(precision=3, suppress=True)
    random.seed(10)

    env = RLEnvironment()
    # env.plot_world()
    print(env.agent_info)  # Output the current agent information

    # Example usage:
    grid_size = 5
    num_actions = 10
    # Initialize the Q-table
    q_table = np.zeros((num_actions, grid_size, grid_size))

    # Set hyperparameters
    alpha = 0.3  # Learning rate
    gamma = 0.5  # Discount factor
    epsilon = 0.1 # Exploration vs Exploitation factor

    num_steps = 1000
    PRANDOM(500, env, q_table, alpha, gamma, epsilon, 'random', 'q-learning')
    print(q_table)


if __name__ == "__main__":
    main()
