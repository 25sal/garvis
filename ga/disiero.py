from deap import base, creator, tools
from deap.tools.emo import assignCrowdingDist
from ga.disiero_functions import crossover_individuals, combined_mutation, create_individual, evaluate, random_seed, population_from_seed,mutate_individual, mutate_swap_segments
from scenario.scenario2d import leggi_dati_csv
from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import random
import time
import csv
import math
from myutils.geometry2d import is_valid_path, distance
import numpy as np
from scenario.conf import scn
import sys
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Parametri globali ---
area_size = scn["area_size"]
speed_min = scn["speed_min"]
speed_max = scn["speed_max"]
separation_min = scn["separation_min"]
n_experiments = scn["n_experiments"]
first_experiment = scn["first_experiment"]
pop_size = scn["pop_size"]
n_generations = scn["n_generations"]
n_segments = scn["n_segments"]
sampling_interval = scn["sampling_interval"]
KMH_TO_NM_MIN = scn["KMH_TO_NM_MIN"]
input_scenarios_file = scn["input_scenarios_file"]


class Scenario():
    def __init__(self, ing_early, usc_early, p_inc, speed_early, ing_late, usc_late, speed_late, area_size, sampling_interval=30, separation_min=2.5):
        self.ing_early = ing_early
        self.usc_early = usc_early
        self.p_inc = p_inc
        self.speed_early = speed_early
        self.ing_late = ing_late
        self.usc_late = usc_late
        self.speed_late = speed_late
        self.area_size = area_size
        self.sampling_interval = sampling_interval
        self.separation_min = separation_min

random.seed(42)  # Per riproducibilità

# DEAP Setup
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("mate", crossover_individuals)
toolbox.register("mutate", combined_mutation)
toolbox.register("select", tools.selNSGA2)


# ---------------------------------------------------
# Simulazione
# ---------------------------------------------------
area_poly = Polygon([(0, 0), (0, area_size), (area_size, area_size), (area_size, 0)])
aerei_data, collision_points = leggi_dati_csv(input_scenarios_file)

for exp in range(n_experiments):
    logger.info(f"Running experiment {exp + 1}/{n_experiments}.")

    p_inc = collision_points[exp]
    aerei = aerei_data[exp*2:exp*2+2]
    

    a1, a2 = aerei
    t1 = distance(a1[0], a1[2]) / a1[3]
    t2 = distance(a2[0], a2[2]) / a2[3]
    early, late = (a1, a2) if t1 < t2 else (a2, a1)

    ing_early, usc_early, p_inc, speed_early = early
    ing_late, usc_late, _, speed_late = late
    t_col = distance(ing_early, p_inc) / speed_early
    traj_early = LineString([ing_early, usc_early])
    scn["ing_late"] = ing_late
    scn["usc_late"] = usc_late
    scn["ing_early"] = ing_early
    scn["usc_early"] = usc_early
    scn["speed_late"] = speed_late
    scn["speed_early"] = speed_early

    original_line = LineString([ing_late, usc_late])

    seed = random_seed(scn, n_segments)
    logging.info(f" Generating initial population with seed: {seed}")
    population = population_from_seed(pop_size, n_segments, scn=scn, seed=seed)
    logger.info(f" Popolazione iniziale generata con seed: {seed}, {len(population)} individui validi.")

    # Qui registri evaluate con TUTTI gli argomenti
    toolbox.register("evaluate", lambda ind: evaluate(ind, usc_late, original_line))


    # Assegno la fitness
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # Probabilità più basse per crossover e mutazione
    CROSSOVER_PROB = 0.6
    MUTATION_PROB = 0.2

    start_total = time.time()  # Timer totale

    for gen in range(n_generations):
        start_gen = time.time()
        offspring = []
        scartati = 0  # Contatore per figli non validi

        while len(offspring) < pop_size:
            parent1, parent2 = toolbox.select(population, 2)
            child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)

            if random.random() < CROSSOVER_PROB:
                child1, child2 = toolbox.mate(child1, child2)

            if random.random() < MUTATION_PROB:
                child1, = toolbox.mutate(child1)
            if random.random() < MUTATION_PROB:
                child2, = toolbox.mutate(child2)

            for child in (child1, child2):
                if is_valid_path(child, scn):
                    child.fitness.values = toolbox.evaluate(child)
                    offspring.append(child)
                else:
                    scartati += 1

            combined = population + offspring

        # Selezione NSGA-II
        pareto_front = tools.sortNondominated(combined, len(combined))
        for front in pareto_front:
            assignCrowdingDist(front)

        population = tools.selNSGA2(combined, pop_size)

        # Log della generazione
        elapsed_gen = time.time() - start_gen
        logger.info(f"⏳ Generazione {gen+1}/{n_generations} completata in {elapsed_gen:.2f}s – Scartati: {scartati}")
  
    
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    logger.info(f" Numero di traiettorie non dominate: {len(pareto_front)}")    
    with open(f"data/pareto_scenario_{exp}.json", "w", newline="") as file:
        result = {}
        
        result['scenario']= {
            'early': {
            'ingress': ing_early,
            'exit': usc_early,
            'speed': speed_early
        },
            'late': {
                'ingress': ing_late,
                'exit': usc_late,
                'speed': speed_late
            },
            'collision_point': p_inc,
            'area_size': area_size
        }
        result['seed'] = seed
        result['pareto_front'] =[]
        for ind in pareto_front:
            result['pareto_front'].append( {
                'path': ind,
                'fitness': ind.fitness.values
            })
        
        json.dump(result, file, indent=4)
        file.close()
        