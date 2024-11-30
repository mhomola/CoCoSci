# Pedestrian Goal Inference for Autonomous Vehicles

## Overview
This 9.660 project implements a Bayesian inverse planning model to infer pedestrian goals and make autonomous vehicle decisions in an urban environment. The model observes partial pedestrian trajectories and infers the probability distribution over possible destinations, considering physical constraints like obstacles and the road layout.

## Method

### Grid World Environment
7x7 grid world with three possible destinations: Arena, Bank, and Cafe; road runs horizontally through the middle; contains obstacles that pedestrians must avoid; car travels along the road and must decide whether to slow down based on pedestrian behavior

### Bayesian Inference Model

#### 1. Trajectory Likelihood Calculation
The likelihood of a trajectory given a goal is calculated using:

```
python
P(trajectory | goal) ∝ exp(-β * (total_cost - optimal_cost)) * exp(β * progress)
```

where:
- `total_cost`: Actual path cost + remaining distance to goal
- `optimal_cost`: Shortest possible path length from start to goal avoiding obstacles
- `progress`: How much closer the trajectory gets to the goal
- `β`: Rationality parameter (higher values = more rational behavior)

The likelihood incorporates:
- Obstacle avoidance using A* pathfinding
- Path optimality (shorter paths are more likely)
- Goal-directed progress
- Physical constraints of the environment

#### 2. Goal Inference
Uses Bayes' rule to compute goal probabilities:

```python
P(goal | trajectory) ∝ P(trajectory | goal) * P(goal)
```

where:
- `P(goal)`: Prior probability of each goal (uniform by default)
- `P(trajectory | goal)`: Likelihood from trajectory analysis
- Probabilities are normalized to sum to 1

#### 3. Decision Making
The car's decision to slow down is based on:
1. Probability of goals requiring road crossing
2. Crossing probability threshold (default set to 0.3)

### Implementation Details

#### A* Pathfinding
Used to find optimal paths avoiding obstacles:
1. Maintains priority queue based on f(n) = g(n) + h(n)
2. g(n): Actual path cost from start
3. h(n): Manhattan distance heuristic to goal
4. Returns path length or infinity if no valid path exists

#### Trajectory Analysis
For each observed trajectory:
1. Checks for obstacle collisions
2. Calculates path costs considering obstacles
3. Computes progress toward goals
4. Evaluates rationality of movement

## Usage

```
python
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
```

## Parameters

- `beta`: Controls rationality assumption (default 0.5)
  - Higher values → more rational behavior expected
  - Lower values → more random behavior allowed
- `DECISION_THRESHOLD`: Threshold for car slowdown (default 0.3)
  - Higher values → more conservative driving
  - Lower values → more aggressive driving

## References

Based on the Bayesian Theory of Mind framework from:
- Baker, Jara-Ettinger, Saxe, Tenenbaum, "Rational quantitative attribution of beliefs, desires and percepts in human mentalizing", Nature Human Behavior, 2017
- Jara-Ettinger et al., "The Naïve Utility Calculus: Computational Principles Underlying Commonsense Psychology", Trends in Cognitive Sciences, 2016
