import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def manhattan_distance(pos1, pos2):
    """Man distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_valid_move(pos, obstacles, grid_size):
    """Is the position valid? Not in obstacle."""
    x, y = pos
    return (0 <= x < grid_size[0] and
            0 <= y < grid_size[1] and
            pos not in obstacles)


def find_shortest_path(start, goal, obstacles, grid_size):
    """
    Findthe shortest path between start and goal avoiding obstacles (using A*).
    """

    def heuristic(pos):
        return manhattan_distance(pos, goal)

    from heapq import heappush, heappop
    queue = [(0 + heuristic(start), 0, start, [start])]  # priority queue
    visited = set()

    while queue:
        _, cost, current, path = heappop(queue)

        if current == goal:
            return len(path) - 1  # num of steps (edges) in path

        if current in visited:
            continue

        visited.add(current)

        # try all possible moves
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in moves:
            next_pos = (current[0] + dx, current[1] + dy)
            if is_valid_move(next_pos, obstacles, grid_size):
                new_cost = cost + 1
                heappush(queue, (new_cost + heuristic(next_pos),
                                 new_cost, next_pos, path + [next_pos]))

    return float('inf')  # no path found :(


def trajectory_likelihood(goal, trajectory, obstacles, grid_size, beta=0.5):
    """
    Calculate the likelihood of a trajectory given a goal
    --> paths closer to optimal are more likely.
    """

    if not trajectory:
        return 1.0

    # calc the cost of observed path while checking for obstacle collisions
    path_cost = 0
    for i in range(len(trajectory) - 1):
        if trajectory[i] in obstacles or trajectory[i + 1] in obstacles:
            return 0.0  # zero lklhood for paths through obstacles
        step_cost = manhattan_distance(trajectory[i], trajectory[i + 1])
        path_cost += step_cost

    # calc the optimal path length (considering obstacles)
    optimal_cost = find_shortest_path(trajectory[0], goal, obstacles, grid_size)
    if optimal_cost == float('inf'):
        return 0.0  # Zero likelihood if no valid path exists

    # calc remaining path length (considering obstacles)
    remaining_cost = find_shortest_path(trajectory[-1], goal, obstacles, grid_size)
    if remaining_cost == float('inf'):
        return 0.0  # zero likelihood if no valid path exists

    total_cost = path_cost + remaining_cost

    # higher likelihood for paths closer to optimal
    likelihood = np.exp(-beta * (total_cost - optimal_cost))

    # Also reward paths that make progress toward goal
    start_to_goal = find_shortest_path(trajectory[0], goal, obstacles, grid_size)
    current_to_goal = find_shortest_path(trajectory[-1], goal, obstacles, grid_size)
    if start_to_goal == float('inf') or current_to_goal == float('inf'):
        progress = 0
    else:
        progress = (start_to_goal - current_to_goal) / len(trajectory)
    progress_factor = np.exp(beta * progress)
    progress_factor = 1
    return likelihood * progress_factor


def infer_goal_and_decide(grid, pedestrian_start, pedestrian_goals, observed_trajectory, car_path, obstacle_cells):
    """
    Infer pedestrian's goal and decide car behavior (Bayesian inference)
    """

    # Dictionary of goals with equal priors
    goal_names = ['Arena', 'Bank', 'Cafe']
    goal_dict = dict(zip(goal_names, pedestrian_goals))
    num_goals = len(pedestrian_goals)
    prior = 1.0 / num_goals

    # Calculate likelihoods for each goal (considering obstacles)
    grid_size = grid.shape
    likelihoods = {}
    for name, goal_pos in goal_dict.items():
        likelihoods[name] = trajectory_likelihood(goal_pos, observed_trajectory,
                                                  obstacle_cells, grid_size)

    # Calculate posteriors
    posteriors = {}
    total_probability = 0
    for name in goal_dict:
        posteriors[name] = likelihoods[name] * prior
        total_probability += posteriors[name]

    # Handle zero probability case
    if total_probability == 0:
        print("Warning: Total probability is zero. Invalid trajectory or model parameters.")
        goal_probabilities = {name: 1.0 / num_goals for name in goal_dict}  # Assign uniform probabilities
        return goal_probabilities, "Continue"

    # Normalize probabilities
    goal_probabilities = {name: prob / total_probability
                          for name, prob in posteriors.items()}

    # Car behavior
    crossing_probability = 0
    road_y = 3  # Assuming road is at y=3

    current_pos = observed_trajectory[-1]
    for name, goal_pos in goal_dict.items():
        # Check if goal requires crossing
        if (current_pos[0] > road_y and goal_pos[0] < road_y) or \
                (current_pos[0] < road_y and goal_pos[0] > road_y):
            crossing_probability += goal_probabilities[name]

    DECISION_THRESHOLD = 0.9  # We can tune this!
    car_decision = "Slow Down" if crossing_probability > DECISION_THRESHOLD else "Continue"

    return goal_probabilities, car_decision


def add_icon(ax, position, icon_path, zoom=0.1):
    """Add an icon to the plot with adjustable size."""
    image = plt.imread(icon_path)
    imagebox = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(imagebox, (position[1], position[0]), frameon=False)
    ax.add_artist(ab)


def visualize(grid, pedestrian_start, pedestrian_goals, observed_trajectories, car_path,
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
    for trajectory in observed_trajectories:
        ped_x = [pos[1] for pos in trajectory]
        ped_y = [pos[0] for pos in trajectory]
        ax.plot(ped_x, ped_y, c='blue', label="Pedestrian Trajectory", linewidth=2, alpha=0.5)
        ax.scatter(ped_x, ped_y, c='blue', s=50, alpha=0.5)

    # pedestrian
    add_icon(ax, pedestrian_start, pedestrian_image_path, zoom=0.0275)

    # goals
    goal_names = ["Arena", "Bank", "Cafe"]
    for goal, name in zip(pedestrian_goals, goal_names):
        prob = goal_probabilities[name]
        ax.add_patch(plt.Rectangle((goal[1] - 0.5, goal[0] - 0.5), 1, 1, color="brown"))
        ax.text(goal[1], goal[0], f"{name}\n", color="white", fontsize=10, ha="center", va="center")
        #ax.text(goal[1], goal[0], f"{name}\n{prob:.2f}", color="white", fontsize=10,
        #        ha="center", va="center")

    # carr
    add_icon(ax, car_path[0], car_image_path, zoom=0.05)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":

    # define scenario
    grid = np.zeros((7, 7))
    pedestrian_start = (6, 2)
    pedestrian_goals = [(1, 1), (1, 5), (5, 6)]  # Arena, Bank, Cafe

    # Different test trajectories
    trajectories = {
        "toward_arena":         [(6, 2), (5, 2), (5, 1)],
        "toward_bank":          [(6, 2), (5, 3), (4, 4), (3, 5)],
        "toward_cafe":          [(6, 2), (6, 3), (5, 4)],
        "obstacle_avoidance":   [(6, 2), (5, 1), (4, 1)],
        "Path A":               [(6, 2), (5, 2), (5, 1), (4, 1)],
        "Path B":               [(4, 3), (4, 2)], #(4, 1)],
        "Path C":               [(2, 5), (2, 6)],
        "Path D":               [(6, 2), (6, 3), (6, 4)]
    }

    observed_trajectories = [trajectories["Path B"]]  # change to test different trajectory
    car_path = [(3, i) for i in range(7)]
    obstacle_cells = [(2, 2), (2, 3), (2, 4), (4, 4), (4, 5)]

    # inference (not sure if correct, gotta check!)
    goal_probabilities, car_decision = infer_goal_and_decide(
        grid, pedestrian_start, pedestrian_goals, observed_trajectories[0], car_path, obstacle_cells)

    #print("Goal Probabilities:", goal_probabilities)
    formatted_probabilities = ", ".join([f"{goal}: {float(prob):.4f}" for goal, prob in goal_probabilities.items()])
    print("Goal Probabilities:", formatted_probabilities)
    print("Car Decision:", car_decision)

    car_image_path = "car_icon.png"
    pedestrian_image_path = "pedestrian_icon.png"

    visualize(
        grid,
        observed_trajectories[0][-1],
        pedestrian_goals,
        observed_trajectories,
        car_path,
        goal_probabilities,
        car_decision,
        obstacle_cells,
        car_image_path,
        pedestrian_image_path
    )

