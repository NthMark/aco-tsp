import numpy as np

def getTspData(tsp):
    # Open input file
    infile = open(tsp, 'r')

    # Read instance
    name = infile.readline().strip().split()[1]                     # NAME
    type = infile.readline().strip().split()[1]                     # TYPE
    comment = infile.readline().strip().split()[1]                  # COMMENT
    dimension = infile.readline().strip().split()[1]                # DIMENSION
    edge_weight_type = infile.readline().strip().split()[1]         # EDGE_WEIGHT_TYPE
    node_coord_section = []                                         # NODE_COORD_SECTION
    infile.readline()

    # Read node coord section and store its x, y coordinates
    for _ in range(0, int(dimension)):
        x, y = infile.readline().strip().split()[1:]
        node_coord_section.append([float(x), float(y)])

    # Close input file
    infile.close()

    # File as dictionary
    return {
        'name': name,
        'type': type,
        'comment': comment,
        'dimension': dimension,
        'edge_weight_type': edge_weight_type,
        'node_coord_section': node_coord_section
    }

def displayTspHeaders(dict):
    print('\nName: ', dict['name'])
    print('Type: ', dict['type'])
    print('Comment: ', dict['comment'])
    print('Dimension: ', dict['dimension'])
    print('Edge Weight Type: ', dict['edge_weight_type'], '\n')

def runAcoTsp(space, iterations = 80, colony = 50, alpha = 1.0, beta = 1.0, Q = 5.0, rho = 0.5):
    # Find inverted distances for all nodes
    inv_distances = inverseDistances(space)

    # Add beta algorithm parameter to inverted distances
    inv_distances = inv_distances ** beta

    # Empty pheromones trail
    pheromones = np.zeros((space.shape[0], space.shape[0]))
    distance_diagram=list()
    # Empty minimum distance and path
    min_distance = None
    min_path = None

    # For the number of iterations
    for i in range(iterations):
        # Initial random positions
        positions = initializeAnts(space, colony)

        # Complete a path
        paths,delta_pheromones = moveAnts(space, positions, inv_distances, pheromones, alpha, beta, Q)

        # Evaporate pheromones
        pheromones = (1 - rho)*pheromones+delta_pheromones

        # [3] For each path
        for path in paths:
            # Empty distance
            distance = 0

            # For each node from second to last
            for node in range(1, path.shape[0]):
                # Calculate distance to the last node
                distance += np.sqrt(((space[int(path[node])] - space[int(path[node - 1])]) ** 2).sum())

            # Update minimun distance and path if less nor non existent
            if not min_distance or distance < min_distance:
                min_distance = distance
                min_path = path
        distance_diagram.append((i,min_distance))
    return (min_path, min_distance,distance_diagram)

def inverseDistances(space):
    distances = np.zeros((space.shape[0], space.shape[0]))

    for index, point in enumerate(space):
        distances[index] = np.sqrt(((space - point) ** 2).sum(axis = 1))

    with np.errstate(all = 'ignore'):
        inv_distances = 1 / distances

    inv_distances[inv_distances == np.inf] = 0

    return inv_distances

def initializeAnts(space, colony):
    # Indexes of initial positions of ants
    return np.random.randint(space.shape[0], size = colony)

def moveAnts(space, positions, inv_distances, pheromones, alpha, beta, Q):
    # Empty multidimensional array (matriz) to paths
    paths = np.zeros((space.shape[0], positions.shape[0]), dtype = int) - 1
    delta_pheromones= np.zeros((space.shape[0], space.shape[0]), dtype = float)
    # Initial position at node zero
    paths[0] = positions

    # For nodes after start to end
    for node in range(1, space.shape[0]):
        # For each ant
        for ant in range(positions.shape[0]):
            prev_node=paths[node-1,ant]
            # Probability to travel the nodes
            next_location_probability = (inv_distances[prev_node] ** beta + pheromones[prev_node] ** alpha) /(inv_distances[prev_node].sum() ** beta + pheromones[prev_node].sum() ** alpha)

            # Index to maximum probability node
            next_position = np.argwhere(next_location_probability == np.amax(next_location_probability))[0][0]

            # Check if node has already been visited
            while next_position in paths[:, ant]:
                # Replace the probability of visited to zero
                next_location_probability[next_position] = 0.0

                # Find the maximum probability node
                next_position = np.argwhere(next_location_probability == np.amax(next_location_probability))[0][0]

            # Add node to path
            paths[node, ant] = next_position

    # Update pheromones (releasing pheromones)
    paths=paths.T
    for path in paths:
        distance = 0
        for node in range(1, path.shape[0]):
            distance += np.sqrt(((space[int(path[node])] - space[int(path[node - 1])]) ** 2).sum())
        for node in range(1, path.shape[0]):
            delta_pheromones[int(path[node-1]),int(path[node])]+=Q/distance

    # Paths taken by the ants
    return paths,delta_pheromones
