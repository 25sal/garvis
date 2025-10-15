
import random
from deap import base, creator, tools, algorithms
import deap_ga_module_v2 as ga
from scenario.conf import scn
import logging
from shapely.geometry import LineString, Polygon
from myutils.geometry2d import distance
from myutils.geometry2d import is_valid_path
from scenario.scenario2d import leggi_dati_csv
from myutils.geometry2d import compute_initial_bearing
import matplotlib.pyplot as plt
import math
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Parametri globali ---
area_size = scn["area_size"]
separation_min = scn["separation_min"]
n_experiments = scn["n_experiments"]
first_experiment = scn["first_experiment"]
sampling_interval = scn["sampling_interval"]
KMH_TO_NM_MIN = scn["KMH_TO_NM_MIN"]
input_scenarios_file = scn["input_scenarios_file"]


# Parametri evolutivi
POP_SIZE = scn["pop_size"]
NGEN = scn["n_generations"]
CXPB = 0.7
MUTPB = 0.3
NUM_GENI = scn["n_segments"]-1



def init_ind():
    return creator.Individual(ga.init_individuo_random(num_geni=NUM_GENI, speed_lim=(scn["speed_min"]*KMH_TO_NM_MIN, scn    ["speed_max"]*KMH_TO_NM_MIN), theta_lim=scn["max_turn_angle_deg"], time_span=(0, scn["MAX_TIME"])))

def evaluate(ind, exp_scenario):
    fit, sim = ga.valuta_individuo(ind, exp_scenario)
    ind.sim = sim
    return fit

def draw_population(population, exp_scenario, filename="population.png"):
    plt.figure(figsize=(10, 7))
    for ind in population:
        x = [p[0] for p in ind.sim["path"]]
        y = [p[1] for p in ind.sim["path"]]
        plt.plot(x, y, color='b', alpha=0.3)
    plt.plot([exp_scenario["ing_late"][0], exp_scenario["usc_late"][0]], [exp_scenario["ing_late"][1], exp_scenario["usc_late"][1]], color='orange', label='Late UAV')
    plt.plot([exp_scenario["ing_early"][0], exp_scenario["usc_early"][0]], [exp_scenario["ing_early"][1], exp_scenario["usc_early"][1]], color='g', label='Early UAV')
    circle = plt.Circle(exp_scenario["collision_point"], 2.5, color='r', fill=True)
    plt.gca().add_artist(circle)
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.xlabel("X (NM)")
    plt.ylabel("Y (NM)")
    plt.title("Population Trajectories")
    plt.savefig(filename)
    plt.close()

def main():
    
    area_poly = Polygon([(0, 0), (0, area_size), (area_size, area_size), (area_size, 0)])
    aerei_data, collision_points = leggi_dati_csv(input_scenarios_file)

    # Definizione tipo fitness con 3 obiettivi: massimizza primo, minimizza secondo e terzo
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti, sim=dict)

    for exp in range(n_experiments):
        p_inc = collision_points[exp]
        aerei = aerei_data[exp*2:exp*2+2]
        a1, a2 = aerei
        t1 = distance(a1[0], a1[2]) / a1[3]
        t2 = distance(a2[0], a2[2]) / a2[3]
        early, late = (a1, a2) if t1 < t2 else (a2, a1)

        ing_early, usc_early, p_inc, speed_early = early
        ing_late, usc_late, _, speed_late = late
        traj_early = LineString([ing_early, usc_early])
        scn["ing_late"] = ing_late
        scn["usc_late"] = usc_late
        scn["ing_early"] = ing_early
        scn["usc_early"] = usc_early
        scn["speed_late"] = speed_late
        scn["speed_early"] = speed_early
       
        # max_time is the time to collision
        collision_time = 60 * distance(ing_early, p_inc) / speed_early
        exit_late_time = 60 * distance(ing_late, usc_late) / speed_late
        
        exp_scenario = {
            "ing_early": ing_early,
            "usc_early": usc_early,
            "speed_early": speed_early,
            "ing_late": ing_late,
            "usc_late": usc_late,
            "speed_late": speed_late,
            "exit_late_time": exit_late_time,
            "collision_point": p_inc,
            "collision_time": collision_time
        }
       
        scn["MAX_TIME"] = exit_late_time 
       
        toolbox = base.Toolbox()
        toolbox.register("individual", init_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", ga.crossover_un_punto)
        toolbox.register("mutate", ga.mutazione_gene)
        toolbox.register("select", tools.selNSGA2)
        pop = toolbox.population(n=POP_SIZE)


        logger.info(f"--- Experiment {exp+1}/{n_experiments} with entry {ing_early}, exit {usc_early}, speed {speed_early} ---")
        logger.info(f"collision_time: {collision_time} min, exit_time_late: {distance(ing_late, usc_late)/speed_late} min \n ")
        logger.info(f"--- Starting population {exp+1}/{n_experiments} with entry {ing_late}, exit {usc_late}, speed {speed_late} ---")


         
        # Inizializza fitness
        
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind, exp_scenario)

        draw_population(pop, exp_scenario, filename=f"data/{exp}_starting_population.png")
      
  
        tools.emo.assignCrowdingDist(pop)
        for gen in range(NGEN):
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    c1, c2 = toolbox.mate(ind1, ind2)
                    ind1[:] = c1
                    ind2[:] = c2

            for ind in offspring:
                if random.random() < MUTPB:
                    ind[:] = toolbox.mutate(ind)

            for ind in offspring:
                ind.fitness.values = toolbox.evaluate(ind,exp_scenario)

            tools.emo.assignCrowdingDist(offspring)
            pop = toolbox.select(pop + offspring, POP_SIZE)

        non_dominated = tools.sortNondominated(pop, len(pop), True)[0]
        logger.info(f"--- Non dominated solutions: {len(non_dominated)} ---")
        collisions = 0
        from myutils.geometry2d import collision_detection
        for ind in non_dominated:
            if collision_detection(ind.sim["path"], p_inc, separation_min, [collision_time-2.5*60/speed_early, collision_time+2.5*60/speed_early]):
                collisions +=1
        logger.info(f"--- Collisions in non dominated solutions: {collisions} ---")
        ga.save_pareto_front(exp_scenario, tools.sortNondominated(pop, len(pop), True)[0], f"data/{exp}_pareto_front.json")

if __name__ == "__main__":
    main()
