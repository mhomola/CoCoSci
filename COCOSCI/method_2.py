import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

# MAKE A LEARNED POLICY FOR THE HUMAN (Q FUNCTION) AND USE THE POLICY TO INFORM THE LIKELIHOOD

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

def ped_reward(current_state, next_state, obstacle_cells, goal_state, car_path):
    ''' Reward for pedestrian - based on distance to goal w/ obstacle avoidance'''
    current_distance = manhattan_distance(current_state, goal_state)
    next_distance = manhattan_distance(next_state, goal_state)
    
    if next_state == goal_state:
        return 50
    elif next_state in obstacle_cells:
        return -50
    elif next_state in car_path:
        return -2 # small reward for walking in road
    else:
        # negative reward for moving away from goal, positive for moving closer
        # could also implement something that discounts with each step
        return current_distance-next_distance
    
    
def compute_similarity(observed_trajectory, ped_traj, penalty=0.3):
    ''' Compare observed trajectory with predicted path '''
    similarity = 0
    
    #print(observed_trajectory)
    for obs, pred in zip(observed_trajectory, ped_traj):
        obs_state, obs_action = obs
        pred_state, pred_action = pred

        if obs_state == pred_state:
            if obs_action == pred_action:
                similarity += 1
        else:
            #similarity -= np.exp(beta * np.abs(obs_action - pred_action))
            obs_state = np.array(obs_state)
            pred_state = np.array(pred_state)
            similarity -= np.exp(penalty * np.linalg.norm(obs_state - pred_state)) 
    
    return similarity 

def softmax(goal_similarities, beta=0.7):
    ''' Compute the softmax over the goal similarities '''
    exp_similarities = np.exp(beta* np.array(goal_similarities))
    softmax_probs = exp_similarities / np.sum(exp_similarities)
    return softmax_probs

def learn_ped_path(goal_state, observed_trajectory, alpha=0.8, gamma=0.9, epsilon=0.2, epochs=1000):
    ''' Q-learning to determine ped path '''
    obstacle_cells = [(2, 2), (2, 3), (2, 4), (4, 4), (4, 5)]
    ped_start = observed_trajectory[-1][0] # most recent obs
    n_ped_states = 49 # size of grid
    n_ped_actions = 4 # (up, down, left, right) 
    actions = [0, 1, 2, 3] #['up', 'down', 'left', 'right']
    car_path = [(3, i) for i in range(7)]
    Q_table_ped = np.zeros((n_ped_states, n_ped_actions))
   
    for epoch in range(epochs):
        current_state = ped_start
        obs = []
        
        while current_state!= goal_state:
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

            # Reward function
            reward = ped_reward(current_state, next_state, obstacle_cells, goal_state, car_path)

    
            # Update Q-value using the Q-learning update rule - Bellman eqn
            next_state_idx = loc_to_idx(next_state)

            Q_table_ped[current_state_idx, ped_action] += alpha * (reward + gamma * np.max(Q_table_ped[next_state_idx]) - Q_table_ped[current_state_idx, ped_action])

            
            current_state = next_state  # Move to the next state


    return Q_table_ped

def predict_ped_path(Q_table_ped, start_state, goal_state, steps=10):
    ''' Predicts pedestrian path using Q-values by applying policy to each time step '''
    current_state = start_state
    traj = [(current_state, None)]
    
    for _ in range(steps):
        state_idx = loc_to_idx(current_state)
        best_action = np.argmax(Q_table_ped[state_idx]) # find best next action from Q-vals - extract policy
        current_state = step(current_state, best_action)
        traj.append((current_state, best_action))
        
        if current_state == goal_state:
            break
        
    return traj

def get_goal_likelihoods_from_q(observed_trajectory, goal_states, beta=0.7): #high beta=deterministic, low beta=random
    ''' Use Q-values to obtain goal likelihoods by computing similarities between trajectories and taking the softmax'''    
    Q_tables = []
    goal_sims = []
    for goal_state in goal_states:
        Q_table = learn_ped_path(goal_state, observed_trajectory, alpha=0.8, gamma=0.9, epsilon=0.2, epochs=1000)
        Q_tables.append(Q_table)
        ped_traj = predict_ped_path(Q_table, observed_trajectory[0][0], goal_state)
        
        sim = compute_similarity(observed_trajectory, ped_traj)
        goal_sims.append(sim)
   
    get_softmax = softmax(goal_sims)
    goals = ['Arena', 'Bank', 'Cafe']
    goal_likelihoods = {goals[i]: get_softmax[i] for i in range(len(get_softmax))}
    return goal_likelihoods


def car_decision_from_ped_updated(goal_likelihoods, observed_trajectory):
    ''' Make action decision based on goal likelihoods and crossing threshold'''
    best_goal_name = max(zip(goal_likelihoods.values(), goal_likelihoods.keys()))[1]
    pedestrian_goals = {'Arena': (1, 1), 'Bank': (1, 5), 'Cafe': (5, 6)}
    best_goal = pedestrian_goals[best_goal_name]
    
    crossing_probability = 0
    current_pos = observed_trajectory[-1][0]
    #print(current_pos, best_goal)
    road_x = 3 # road location
    if (current_pos[0] > road_x and best_goal[0] < road_x) or (current_pos[0] < road_x and best_goal[0] > road_x):
        crossing_probability += goal_likelihoods[best_goal_name]
    
    if crossing_probability > 0.4: #can change thsi
        car_action = 'Slow down'
    else:
        car_action = 'Continue'
        
    return crossing_probability, best_goal_name, car_action

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
    ped_x = [pos[0][1] for pos in observed_trajectory]
    ped_y = [pos[0][0] for pos in observed_trajectory]
    ax.plot(ped_x, ped_y, c='blue', label="Pedestrian Trajectory", linewidth=2, alpha=0.5)
    ax.scatter(ped_x, ped_y, c='blue', s=50, alpha=0.5)

    # pedestrian
    add_icon(ax, observed_trajectory[-1][0], pedestrian_image_path, zoom=0.0275) #add_icon(ax, pedestrian_start, pedestrian_image_path, zoom=0.0275)

    # goals
    goal_names = ["Arena", "Bank", "Cafe"]
    for goal, name in zip(pedestrian_goals, goal_names):
         prob = goal_probabilities[name]
         ax.add_patch(plt.Rectangle((goal[1] - 0.5, goal[0] - 0.5), 1, 1, color="brown"))
         ax.text(goal[1], goal[0], f"{name}", color="white", fontsize=10,
                ha="center", va="center")

    # # carr
    add_icon(ax, car_state, car_image_path, zoom=0.05)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
def main():
    grid = np.zeros((7, 7))
    obstacle_cells = [(2, 2), (2, 3), (2, 4), (4, 4), (4, 5)]
  
    pedestrian_goals = {(1, 1): 'Arena', (1, 5): 'Bank', (5, 6): 'Cafe'}  # Arena, Bank, Cafe
    goal_states = [(1, 1), (1, 5), (5, 6)] # arena, bank, cafe
    car_path = [(3, i) for i in range(7)]
    car_image_path = "car_icon.png"
    pedestrian_image_path = "pedestrian_icon2.png"
    
    #trajectory = [((6, 2), None), ((5, 2), 1), ((5, 1), 3), ((4, 1), 1)] #state-action pairs # comment to change trajectories
    #trajectory = [((4,3), None), ((4,2), 3), ((4, 1), 1)]
    trajectory = [((2,5), None), ((2, 6), 3)]
    car_state = car_path[0]
    ped_start = trajectory[0][0]
    observed_trajectory = []
    for i in range(len(trajectory)): #simulate steps
        observed_trajectory.append(trajectory[i])
        goal_likelihoods = get_goal_likelihoods_from_q(observed_trajectory, goal_states)
        print(f'All likelihoods: {goal_likelihoods}')
        goal_probabilities, best_goal, car_decision = car_decision_from_ped_updated(goal_likelihoods, observed_trajectory)
        
        print(f'Goal likelihood: {goal_likelihoods[best_goal]} for goal {best_goal}')
        print(f'Car decision: {car_decision}')
        
        visualize(
            grid,
            ped_start,
            pedestrian_goals,
            observed_trajectory,
            car_path,
            car_state,
            goal_likelihoods,
            car_decision,
            obstacle_cells,
            car_image_path,
            pedestrian_image_path
        )

if __name__ == "__main__":
    main()