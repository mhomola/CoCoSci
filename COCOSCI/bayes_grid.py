import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import random

def manhattan_distance(pos1, pos2):
    """Man distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def prior(b_goal):
    ''' Returns prior (1 = crossing, 0 = not) based on perceived ped goal 
        Feel free to change priors '''
    if b_goal == 0:
        return 0.2 # prob that ped will cross if believed to not cross
    elif b_goal == 1:
        return 0.8 # prob of crossing if believed to cross
    else:
        return 0.5

def likelihood(b_goal, ped_state, car_state, obs):
    ''' Likelihood of observing action based on ped goal 
        P(action | goal) '''
    # consider closeness to road in prediction (normalized) 
    # here, consider proximity to one space in front of the car // feel free to adjust
    print(car_state)
    car_x, car_y = car_state
    ped_x, ped_y = ped_state
    prox_to_car = manhattan_distance(ped_state, (car_x+1, car_y)) 
    # normalize
    max_distance = 5 # this makes sense in my head but it might be wrong
    prox_factor = 1 - min(prox_to_car/max_distance, 1)
    
    prob_cross = 0.0

    if len(obs) > 0:
        last_obs = obs[-1]

        if (last_obs == (1, 0) and ped_x < 3) or (last_obs == (-1, 0) and ped_x > 3): # moving toward road
            prob_cross += 0.7 * prox_factor
        elif (last_obs == (1, 0) and ped_x > 3) or (last_obs == (-1, 0) and ped_x < 3): #moving away from road
            prob_cross += 0.1 * prox_factor 
        elif last_obs == (0, 1) or last_obs == (0, -1): # move right or left
            prob_cross += 0.2 * prox_factor # small likelihood of crossing if going left/right

    return prob_cross

def update_belief(state, obs, prior, likelihood):
    ''' Inverse planning portion - updates belief about where pedestrian will move next'''
    car_state, b_goal, ped_state = state
    prob_cross = 1.0
    prob_no_cross = 1.0
    prior_cross = prior(b_goal)
    prior_no_cross = 1 - prior_cross

    for step in obs:
        prob_cross *= likelihood(1, ped_state, car_state, obs)
        #print(prob_cross)
        prob_no_cross *= likelihood(0, ped_state, car_state, obs)
        #print(prob_no_cross)

    # marginal likelihood for obs
    total_prob = prob_cross * prior_cross + prob_no_cross * prior_no_cross    
    
    if total_prob == 0: #this is sketchy
        posterior = prior_cross
    
    else:
        # Bayes it up --> P(cross|obs) = P(obs|cross)P(cross)/P(obs) (double check this bc idk)
        posterior = (prob_cross * prior_cross) / total_prob

    return posterior

def reward(state, action):
    ''' Reward for value iteration
    Input: state {car, belief state of ped, ped}, car action
    Output: reward '''

    car_state, b_goal, ped_state = state
    if car_state == ped_state and b_goal == 0:
        return -1000 # heavy penalty for hitting pedestrian
    elif car_state != ped_state and b_goal == 1:
        return -10 # penalty for stopping when no pedestrian
    elif car_state != ped_state and b_goal == 0:
        return 100 # reward for continuing when no pedestrian
    else:
        return -10 # default small negative reward

def next_move(state, policy):
    ''' Car decision based on probability of person crossing '''
    #x, y = car_state
    #if posterior > 0.5:
    #    action == 1
    #else:
    #    action == 0
    #return action
    #print(policy[state])
    return policy[state]

def transition_func(state, action):
    ''' Simple transition model: For simplicity, assume no transition uncertainty. '''
    car_state, b_goal, ped_state = state
    x, y = car_state
    #next_states = {}
    
    if action == 0 and y < 6:  # Continue
        # Move car forward
        next_state = ((car_state[0], car_state[1] + 1), b_goal, ped_state)
        #next_states[next_state] = 1.0  # deterministic transition
    elif action == 1:  # Stop
        next_state = (car_state, b_goal, ped_state)  # Car stays at the same position
        #next_states[next_state] = 1.0
    else:
        next_state = (car_state, b_goal, ped_state)
    
    return next_state

# low key this is so pointless because it's just choosing between two choices
# can probably use simpler function but it's fine 
def value_iteration(states, actions, reward, transition_func, gamma=0.9, theta=1e-6):
    ''' To find optimal policy - using algo/equation from PADM 2023 notes'''
    #car_states, b_goal, x_peds = states
    
    V = {state: 0 for state in states} # initialize value func
    policy = {state: None for state in states} # initialize policy

    while True:
        delta = 0
        V_new = V.copy()

        for state in states:
            vals = []
            
            for action in actions:
                next_state = transition_func(state, action)
                next_reward = reward(state, action)

                val = next_reward + gamma * V[next_state] # Bellman equation
                #print(V[next_state])
                vals.append(val)
            
            V_new[state] = max(vals)
            policy[state] = actions[np.argmax(vals)]

            # Update delta
            delta = max(delta, abs(V_new[state] - V[state]))
        
        V = V_new

        if delta < theta:
            break # convergnce
    
    return policy, V

def main():
    car_states = [(3, i) for i in range(7)] # position on road
    car_state_obs = [(3,0), (3,1)]
    ped_states = [(6, 2), (6, 3), (5, 4)] # hard code ped traj
    actions = [0, 1] # continue (0) or stop (1) : consider changing to ['stop', 'continue']
    b_goals = [0, 1] # belief state of goal of ped: crossing (1) or not (0)
    b_goal = 0 # initially, believe ped won't cross
    #car_state_obs = car_states[0] #initial pos of car 
    # car_action is in terms of car position (e.g. (3,2)), whereas ped_action is wrt the most recent movement (e.g. (-1, 0))
    
    states = [(car_pos, goal, ped_pos)
            for car_pos in car_states
            for goal in b_goals
            for ped_pos in ped_states]
    obs = []

    for t in range(len(ped_states)):
        car_state = car_state_obs[t]
        ped_state = ped_states[t]
        if t > 0:
            prev_ped_state = ped_states[t-1]
            #ped_state = ped_states[t]
            ped_action = (ped_state[0]-prev_ped_state[0], ped_state[1]-prev_ped_state[1])
            obs.append(ped_action)
        
        state = (car_state, b_goal, ped_state)
        posterior = update_belief(state, obs, prior, likelihood)
        policy, vals = value_iteration(states, actions, reward, transition_func)
        car_action = next_move(state, policy)
        next_car_state, b_goal, throwaway = transition_func(state, car_action)
        car_state_obs.append(next_car_state)
    
        print(f"Time step {t}:")
        #print(f"  Pedestrian state: {ped_state}")
        print(f"  Observations so far: {obs}")
        print(f"  Updated belief (probability of crossing): {posterior}")
        print(f"  Car action (stop = 1, continue = 0): {car_action}")

    #ped_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
   
    #posterior = update_belief(b_goal, ped_state, obs, prior, likelihood)
    #opt_policy, opt_vals = value_iteration(states, actions, reward, next_move)

if __name__ == "__main__":
    main()