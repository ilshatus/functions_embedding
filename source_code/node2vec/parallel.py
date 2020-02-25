import random
import numpy as np
from tqdm import tqdm


def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None,
                            first_travel_probs_key: str = None,
                            first_travel_neighbors_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.

    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            states = [source]

            # Perform walk
            while len(walk) < walk_length:

                if len(states) == 1:  # For the first step
                    walk_options = d_graph[states[-1]].get(first_travel_neighbors_key, None)

                    # Skip dead end nodes
                    if not walk_options:
                        break

                    probabilities = d_graph[states[-1]][first_travel_probs_key]

                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]
                else:
                    walk_options = d_graph[states[-1]].get(neighbors_key, None)

                    # Skip dead end nodes
                    if not walk_options:
                        break

                    walk_options = walk_options.get(states[-2], None)

                    probabilities = d_graph[states[-1]][probabilities_key][states[-2]]
                    walk_to = np.random.choice(walk_options, size=1, p=probabilities)[0]

                    if walk_to == states[-2]:
                        states.pop()
                        states.pop()

                walk.append(walk_to)
                states.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks
