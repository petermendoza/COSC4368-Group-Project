import numpy as np
import random
import matplotlib.pyplot as plt

class RLEnvironment:
    def __init__(self, grid_size=5, num_agents=3):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.grid = np.zeros((grid_size, grid_size), dtype=int)  # Grid initialization
        self.agent_colors = ['red', 'black', 'blue']  # Colors for the agents
        
        # Initialize pickup and dropoff locations
        self.pickup_locations = [(1,5), (2,4), (5,2)]
        self.dropoff_locations = [(1,1), (3,1), (4,5)]
        
        # Initialize agent locations and colors
        self.agent_info = [{'location': (1,3), 'color': 'black'}, 
                           {'location': (3,3), 'color': 'red'}, 
                           {'location': (5,3), 'color': 'blue'}]
        
        # Initialize block counts at pickup and dropoff locations
        self.pickup_blocks = [5] * num_agents
        self.dropoff_blocks = [0] * num_agents
    
    def move_agent(self, agent_id, action):
        """
        Move an agent according to the given action (0: up, 1: down, 2: left, 3: right).
        """
        x, y = self.agent_info[agent_id]['location']
        
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < self.grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < self.grid_size - 1:
            y += 1
        
        # Check if the new position is already occupied by another agent
        for i, agent in enumerate(self.agent_info):
            if i != agent_id and agent['location'] == (x, y):
                return
        
        # Update agent location
        self.agent_info[agent_id]['location'] = (x, y)
        
        # Check if the agent is at a pickup location
        if (x, y) in self.pickup_locations:
            pickup_index = self.pickup_locations.index((x, y))
            # Pick up a block if available
            if self.pickup_blocks[pickup_index] > 0:
                self.pickup_blocks[pickup_index] -= 1
                return 'pickup'
        
        # Check if the agent is at a dropoff location
        if (x, y) in self.dropoff_locations:
            dropoff_index = self.dropoff_locations.index((x, y))
            # Drop off a block if there is space
            if self.dropoff_blocks[dropoff_index] < 5:
                self.dropoff_blocks[dropoff_index] += 1
                return 'dropoff'
        
        return 'move'
    
    def step(self, actions):
        """
        Take a step in the environment given a list of actions for each agent.
        """
        rewards = [0] * self.num_agents
        
        # Move each agent and collect rewards
        for i, action in enumerate(actions):
            reward = self.move_agent(i, action)
            if action == 0 or action == 1 or action == 2 or action == 3:
                rewards[i] -= 1
            
            if reward == 'pickup':
                rewards[i] += 13
            elif reward == 'dropoff':
                rewards[i] += 13
        
        # Check for collisions
        for i, agent1 in enumerate(self.agent_info):
            for j, agent2 in enumerate(self.agent_info):
                if i != j and agent1['location'] == agent2['location']:
                    # Reset agent locations
                    self.agent_info = [{'location': (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)), 'color': color} for color in self.agent_colors]
                    rewards[i] += -1
                    rewards[j] += -1
        
        return rewards, self.agent_info, self.pickup_blocks, self.dropoff_blocks

    def plot_world(self):
        """
        Plot the current state of the world.
        """
        fig, ax = plt.subplots()
        
        # Plot grid
        ax.grid(True)
        ax.set_xticks(np.arange(0, self.grid_size, 1))  # Keep the x-axis ticks as they are
        ax.set_yticks(np.arange(0, self.grid_size, 1))  # Keep the y-axis ticks as they are
        
        # Plot pickup locations
        for loc in self.pickup_locations:
            ax.plot(loc[1], loc[0], color='brown', marker='s', markersize=10)  # Brown square, switch x and y coordinates
        
        # Plot dropoff locations
        for loc in self.dropoff_locations:
            ax.plot(loc[1], loc[0], color='green', marker='s', markersize=10)  # Green square, switch x and y coordinates
        
        # Plot agents
        for agent in self.agent_info:
            color = agent['color']
            loc = agent['location']
            ax.plot(loc[1], loc[0], marker=6, color=color, markersize=10)  # Circle, switch x and y coordinates
        
        ax.set_title('World State')
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Flip the y-axis
        plt.show()




# Example usage:
env = RLEnvironment()
env.plot_world()
print(env.agent_info)  # Output the current agent information
