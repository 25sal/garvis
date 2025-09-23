import random
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from deap import base, creator, tools, algorithms
from deap.tools.emo import assignCrowdingDist
from myutils.geometry2d import generate_random_point, is_valid_path, distance, compute_curvature
import time
from scenario.conf import scn

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def mutate_individual(ind, scn, indpb=0.3):
    """Applica una mutazione sui punti intermedi."""
    for i in range(1, len(ind) - 1):  # Non muovere entry e exit
        if random.random() < indpb:
            x_off = random.uniform(-20, 20)
            y_off = random.uniform(-20, 20)
            ind[i] = (
                min(max(ind[i][0] + x_off, 0), scn["area_size"]),
                min(max(ind[i][1] + y_off, 0), scn["area_size"]),
            )
    return (ind,)

def mutate_swap_segments(ind, prob=0.3):
    """Scambia due punti intermedi della traiettoria con una certa probabilità."""
    if random.random() < prob and len(ind) > 3:
        i, j = sorted(random.sample(range(1, len(ind) - 1), 2))
        ind[i], ind[j] = ind[j], ind[i]
    return (ind,)


def random_seed(scn, n_segments):
    seed = None
    i = 0
    while seed is None:
        logger.debug(f" Generating random seed {i+1}.")
        candidate = create_individual(scn["ing_late"], scn["usc_late"], scn, n_segments)
        if is_valid_path(candidate, scn):
            seed = candidate
        i += 1
    return seed

def population_from_seed(pop_size, n_segments, scn, seed=None):

    if seed is None:
        seed = random_seed(scn, n_segments)

    # Crea la popolazione usando solo la prima seed
    population = []
    while len(population) < pop_size:
        # logger.info(f" Generating individual {len(population)+1}/{pop_size}.")
        mutant = creator.Individual(perturb_seed(seed, scn))
        mutant, = combined_mutation(mutant)
        if is_valid_path(mutant, scn):
            population.append(mutant)
    
    return population

# fitness function
def evaluate(ind, original_exit, original_line):
    path = LineString(ind)

    # Calcola distanza finale
    goal_dist_raw = distance(ind[-1], original_exit)

    # Penalità solo se troppo lontano
    max_allowed_dist = 5.0  # ad esempio, 2 NM
    goal_dist = max(0, goal_dist_raw - max_allowed_dist)

    # Obiettivo 2: fluidità della traiettoria
    curvature = compute_curvature(path)

    # Obiettivo 3: deviazione dalla rotta originale
    avg_dev = np.mean([original_line.distance(Point(x, y)) for x, y in path.coords])

    # Rumore per spezzare simmetrie
    noise = random.uniform(-0.05, 0.05)

    return (goal_dist, -curvature + noise, -avg_dev)

def crossover_individuals(ind1, ind2):
    """Applica crossover a metà dei punti."""
    point = len(ind1) // 2
    new1 = creator.Individual(ind1[:point] + ind2[point:])
    new2 = creator.Individual(ind2[:point] + ind1[point:])
    return new1, new2

def combined_mutation(ind):
    ind, = mutate_individual(ind, scn, indpb=0.9)
    ind, = mutate_swap_segments(ind, prob=0.5)
    return (ind,)

def create_individual(entry, exit_point, scn, n_segments=2):
    """Crea un individuo come lista di punti."""
    coords = [entry]
    for _ in range(n_segments):
        coords.append(generate_random_point(scn["area_size"]))
    coords.append(exit_point)
    return creator.Individual(coords)


def perturb_seed(seed,scn, intensity=15):
    """Applica una variazione casuale ai punti intermedi della seed."""
    perturbed = seed[:]
    for i in range(1, len(seed) - 1):
        x_off = random.uniform(-intensity, intensity)
        y_off = random.uniform(-intensity, intensity)
        perturbed[i] = (
            min(max(perturbed[i][0] + x_off, 0), scn["area_size"]),
            min(max(perturbed[i][1] + y_off, 0), scn["area_size"]),
        )
    return perturbed