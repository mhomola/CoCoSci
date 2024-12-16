import numpy as np
from grid.py import infer_goal_and_decide

# Initialize environment
grid = np.zeros((7, 7))
pedestrian_goals = [(1, 1), (1, 5), (5, 6)]  # Arena, Bank, Cafe
obstacle_cells = [(2, 2), (2, 3), (2, 4), (4, 4), (4, 5)]

# Run inference
goal_probabilities, car_decision = infer_goal_and_decide(
    grid, 
    pedestrian_start,
    pedestrian_goals,
    observed_trajectory,
    car_path,
    obstacle_cells
)