import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

# MAKE A LEARNED POLICY FOR THE HUMAN (Q FUNCTION) AND USE THE POLICY TO INFORM THE LIKELIHOOD
# USE HUMAN POLICY (ASSUMES RATIONAL AGENT) TO INFORM CAR POLICY

def manhattan_distance(pos1, pos2):
    """Man distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def loc_to_idx(state, grid_size=7):
    ''' Convert (x,y) location to index for Q-table'''
    x, y = state
    return x*grid_size + y

def step(current_state, action):
    ''' Defines ped movements for transitioning to next state'''
    x, y = current_state
    if action == 0 and x < 6: #'up' 
        return (x+1, y)
    elif action == 1 and x > 0: #'down'
        return (x-1, y)
    elif action == 2 and y > 0: #" left'
        return (x, y-1)
    elif action == 3 and y < 6: #'right'
        return (x, y+1)
    else:
        return current_state

def ped_reward(current_state, next_state, obstacle_cells, goal_state):
    ''' Reward for pedestrian - based on distance to goal w/ obstacle avoidance'''
    current_distance = manhattan_distance(current_state, goal_state)
    next_distance = manhattan_distance(next_state, goal_state)
    
    if next_state == goal_state:
        return 50
    elif next_state in obstacle_cells:
        return -50
    else:
        # negative reward for moving away from goal, positive for moving closer
        # could also implement something that discounts with each step
        return current_distance-next_distance
    
def car_reward(car_action, car_state, ped_state):
    ''' Reward for car - based on not hitting the pedestrian '''
    if car_action == 'Slow down'  and car_state == ped_state:
        return 100 # pos reward for slowing down
    elif car_action == 'Continue' and car_state == ped_state:
        return -100 # very negative reward for hitting ped
    elif car_action == 'Slow down':
        return -10 # small negative reward for unnecessarily slowing down
    elif car_action == 'Continue':
        return 10 
    else:
        return 0


def learn_ped_path(goal_state, observed_trajectory, alpha=0.8, gamma=0.9, epsilon=0.2, epochs=1000):
    # define environment for ped
    obstacle_cells = [(2, 2), (2, 3), (2, 4), (4, 4), (4, 5)]
    ped_start = observed_trajectory[-1] # only uses most recent observations 
    n_ped_states = 49 # size of grid
    n_ped_actions = 4 # (up, down, left, right) # NO DIAGONAL BC THEN I HAVE TO CHANGE THIS
    actions = [0, 1, 2, 3] #['up', 'down', 'left', 'right']
    Q_table_ped = np.zeros((n_ped_states, n_ped_actions))
   
    for epoch in range(epochs):
        current_state = ped_start
        obs = []
        
        while current_state!= goal_state:
            #print(f'Current state: {current_state} and goal state: {goal_state}')
            current_state_idx = loc_to_idx(current_state)
            
            # Choose action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                ped_action = np.random.choice(actions)  # Explore
            else:
                ped_action = np.argmax(Q_table_ped[current_state_idx])  # Exploit
    
            # Simulate the environment (move to the next state)
            next_state = step(current_state, ped_action)
            
            # Add current obs to trajectory
            obs.append((current_state, ped_action))
            #print(f"Trajectory so far: {obs}")
            # Reward function
            reward = ped_reward(current_state, next_state, obstacle_cells, goal_state)
            #print(f"Reward for current state {current_state}, next state {next_state}: {reward}")
    
            # Update Q-value using the Q-learning update rule - Bellman eqn
            next_state_idx = loc_to_idx(next_state)
            #print(f"Before update, Q-value for state {current_state}: {Q_table_ped[current_state_idx]}")
            Q_table_ped[current_state_idx, ped_action] += alpha * (reward + gamma * np.max(Q_table_ped[next_state_idx]) - Q_table_ped[current_state_idx, ped_action])
            #print(f"After update, Q-value for state {current_state}: {Q_table_ped[current_state_idx]}")
            
            current_state = next_state  # Move to the next state
            #print(f"Updated Q-table for state {current_state}: {Q_table_ped[current_state_idx]}")

    return Q_table_ped

def learn_car_policy(car_state, ped_traj, car_path, crossing_likelihood, alpha=0.8, gamma=0.9, epsilon=0.2, episodes=1000):
    car_actions = ['Slow down', 'Continue']
    goal_state = (3, 6)
    Q_table_car = np.zeros((len(car_path), 2)) #n_car_actions
    for episode in range(episodes):
        current_state = car_path[0] # car start state
        i = 0
        current_state_idx = i
        
        while current_state != goal_state:
            i += 1
            if np.random.rand() < epsilon:
                car_action = random.choice(car_actions) # Explore
            else:
                car_action = car_actions[np.argmax(Q_table_car[current_state_idx])] # Exploit
        
            next_state = car_path[i]
            next_state_idx = current_state_idx+1 # I think
            ped_state = ped_traj[i] # I think ?
            reward = car_reward(car_action, next_state, ped_state) # this aint right
            
            Q_table_car[current_state_idx, car_action] += alpha * (reward + gamma * np.max(Q_table_car[next_state_idx]) - Q_table_car[current_state_idx, car_action])
            
            current_state = next_state
            current_state_idx = next_state_idx

def predict_ped_path(Q_table_ped, start_state, goal_state, steps=10):
    ''' Predicts pedestrian path using Q-values by applying policy to each time step '''
    current_state = start_state
    traj = [current_state]
    
    for _ in range(steps):
        state_idx = loc_to_idx(current_state)
        best_action = np.argmax(Q_table_ped[state_idx]) # find best next action from Q-vals - extract policy
        current_state = step(current_state, best_action)
        traj.append(current_state)
        
        if current_state == goal_state:
            break
    
    return traj
        
def get_goal_likelihoods(ped_traj, goal_states, steps=10):
    ''' Computes Bayesian likelihood of ped heading toward each goal given traj'''
    prior = {goal: 1/3 for goal in goal_states} # equal priors to begin 
    posterior = prior # initialize posterior
    
    likelihoods = {goal: 0 for goal in goal_states} # initialize likelihoods
    for goal in goal_states:
        likelihood = 0
        for step in ped_traj:
            # Calculate likelihood based on proximity to goal?
            distance = manhattan_distance(step, goal)
            step_likelihood = 1 / (1 + distance) # can multiply weight factor by distance if u want
            likelihood += step_likelihood
        
        likelihoods[goal] = likelihood/len(ped_traj) # normalize idk
                
    # Normalize
    total_steps = len(ped_traj)
    total_likelihood = sum(likelihoods.values())
    if total_likelihood == 0:
        return posterior 
    
    likelihoods = {goal: likelihood/total_steps for goal, likelihood in likelihoods.items()}
    # Apply Bayes' theorem: P(Goal | Trajectory) = P(Trajectory | Goal) * P(Goal) / P(Trajectory)
    # For simplicity, P(Trajectory) is just a normalizer (sum over all goals)
    normalizer = sum(likelihoods[goal] * prior[goal] for goal in goal_states)
    
    for goal in goal_states:
        posterior[goal] = (likelihoods[goal]*prior[goal])/ normalizer
    return posterior

def car_decision_from_ped(ped_traj, car_state, car_path, observed_trajectory, goal_states, steps=10):
    ''' Likelihood of ped crossing based on policy 
        Inputs: ped_traj from predict_ped_path fxn, current car pos, road, ped start pos
        Outputs: likelihood of crossing, car action'''
    
    #crossing_likelihood = len(crossing_loc)/len(ped_traj) # based on how many times the ped crosses the road??
    # I hate that
    goal_likelihoods = get_goal_likelihoods(ped_traj, goal_states)
    #print(f'Goal likelihoods: {goal_likelihoods}')
    best_goal = max(zip(goal_likelihoods.values(), goal_likelihoods.keys()))[1]

    crossing_probability = 0
    current_pos = observed_trajectory[-1]
    print(current_pos)
    road_x = 3 # road location
    if (current_pos[0] > road_x and best_goal[0] < road_x) or (current_pos[0] < road_x and best_goal[0] > road_x):
        crossing_probability += goal_likelihoods[best_goal]
    
    if crossing_probability > 0.5: #can change thsi
        car_action = 'Slow down'
    else:
        car_action = 'Continue'
    # predict crossing if goal is on the other side of the street
    # for step in ped_traj:
    #     if step in car_path:
    #         car_action = 'Slow down'
    #         break
    #     else:
    #         car_action = 'Continue'
        
    return goal_likelihoods, best_goal, car_action

def add_icon(ax, position, icon_path, zoom=0.1):
    """Add an icon to the plot with adjustable size."""
    image = plt.imread(icon_path)
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(imagebox, (position[1], position[0]), frameon=False)
    ax.add_artist(ab)


def visualize(grid, pedestrian_start, pedestrian_goals, observed_trajectory, car_path, car_state,
                                  goal_probabilities, car_decision, obstacle_cells, car_image_path,
                                  pedestrian_image_path):
    """
    Visualize the grid-world scenario.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    # grid
    ax.imshow(grid, cmap="Greys", origin="upper")

    # grass
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if row < 3 or row > 3:  # Grass areas
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="lightgreen"))

    # road
    for cell in car_path:
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, color="#C0C0C0"))
    for i in range(7):
        ax.plot([i - 0.5, i + 0.5], [3, 3], 'w--', lw=3)

    # obstacles
    for cell in obstacle_cells:
        ax.add_patch(plt.Rectangle((cell[1] - 0.5, cell[0] - 0.5), 1, 1, color="black"))

    # trajectories
    #for trajectory in observed_trajectories:
    ped_x = [pos[1] for pos in observed_trajectory]
    ped_y = [pos[0] for pos in observed_trajectory]
    ax.plot(ped_x, ped_y, c='blue', label="Pedestrian Trajectory", linewidth=2, alpha=0.5)
    ax.scatter(ped_x, ped_y, c='blue', s=50, alpha=0.5)

    # pedestrian
    add_icon(ax, pedestrian_start, pedestrian_image_path, zoom=0.0275)

    # goals
    goal_names = ["Arena", "Bank", "Cafe"]
    for goal, name in zip(pedestrian_goals, goal_names):
        prob = goal_probabilities[name]
        ax.add_patch(plt.Rectangle((goal[1] - 0.5, goal[0] - 0.5), 1, 1, color="brown"))
        ax.text(goal[1], goal[0], f"{name}\n{prob:.2f}", color="white", fontsize=10,
                ha="center", va="center")

    # carr
    add_icon(ax, car_state, car_image_path, zoom=0.05)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
def main():
    # create grid
    grid = np.zeros((7, 7))
    current_state = (5, 4)
    # ped starts at (6,2) for scenario 1
    # start at (4,3) for immediate crossing prediction scenario 2
    # and another? (7,7) start scenario 3
    obstacle_cells = [(2, 2), (2, 3), (2, 4), (4, 4), (4, 5)]
    #ped_start = (6, 2)
    #n_ped_states = 49 # size of grid
    #n_ped_actions = 4 # (up, down, left, right) # NO DIAGONAL BC THEN I HAVE TO CHANGE THIS
    #actions = [0, 1, 2, 3] #['up', 'down', 'left', 'right']
  
    pedestrian_goals = {(1, 1): 'Arena', (1, 5): 'Bank', (5, 6): 'Cafe'}  # Arena, Bank, Cafe
    goal_states = [(1, 1), (1, 5), (5, 6)] # arena, bank, cafe
    goal_state = goal_states[1] # change to vary
    car_path = [(3, i) for i in range(7)]
    car_image_path = "car_icon.png"
    pedestrian_image_path = "pedestrian_icon2.png"
    # different test trajectories
    # trajectories = {
    #     "toward_arena": [(6, 2), (5, 2), (5, 1)],
    #     "toward_bank": [(6, 2), (5, 3), (4, 3), (3, 5)],
    #     "toward_cafe": [(6, 2), (6, 3), (5, 4)],
    #     "obstacle_avoidance": [(6, 2), (5, 1), (4, 1)]
    # }
    #observed_trajectories = [trajectories["obstacle_avoidance"]]  # change to test different trajectory
    #trajectory = [(6, 2), (5, 2), (5, 1)]
    trajectory = [(4,3)]
    car_state = car_path[0]
    ped_start = trajectory[0]
    observed_trajectory = [trajectory[0]]
    for i in range(len(trajectory)): #simulate three steps
        car_step = 0
        observed_trajectory.append(trajectory[i])
        Q_table = learn_ped_path(goal_state, observed_trajectory)
        #print(f'Q table: {Q_table}')
        #policy = extract_ped_policy(Q_table, goal_state, current_state)
        
        ped_traj = predict_ped_path(Q_table, ped_start, goal_state)
        goal_likelihoods, best_goal, car_decision = car_decision_from_ped(ped_traj, car_state, car_path, observed_trajectory, goal_states)
        goal_probabilities = {'Arena': goal_likelihoods[(1, 1)], 'Bank': goal_likelihoods[(1, 5)], 'Cafe': goal_likelihoods[(5, 6)]}
        print(f'Goal likelihood: {goal_likelihoods[best_goal]} for goal {pedestrian_goals[best_goal]}')
        print(f'Car decision: {car_decision}')
        
        if car_decision == 'Continue':
            car_step += 1
            car_state = car_path[car_step]
        
        visualize(
            grid,
            ped_start,
            pedestrian_goals,
            observed_trajectory,
            car_path,
            car_state,
            goal_probabilities,
            car_decision,
            obstacle_cells,
            car_image_path,
            pedestrian_image_path
        )
    
    # visualize(
    #     grid,
    #     observed_trajectories[0][-1],
    #     pedestrian_goals,
    #     observed_trajectories,
    #     car_path,
    #     policy,
    #     car_decision,
    #     obstacle_cells,
    #     car_image_path,
    #     pedestrian_image_path
    # )
    
if __name__ == "__main__":
    main()