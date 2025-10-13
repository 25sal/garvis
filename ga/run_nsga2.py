
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

def evaluate(ind, p0, theta0, time0, lato, exit_ref):
    fit, sim = ga.valuta_individuo(ind, p0=p0, theta0=theta0, t0=time0, lato=lato, exit_ref=exit_ref)
    ind.sim = sim
    return fit


def main():
    
    area_poly = Polygon([(0, 0), (0, area_size), (area_size, area_size), (area_size, 0)])
    aerei_data, collision_points = leggi_dati_csv(input_scenarios_file)

    # Definizione tipo fitness con 3 obiettivi: massimizza primo, minimizza secondo e terzo
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -1.0))
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
        THETA_0 = 360 * compute_initial_bearing(ing_late, usc_late)/(2*math.pi)
        TIME_0 = 0.0
        # max_time is the time to collision
        t_col = 60 * distance(ing_early, p_inc) / speed_early
       
        scn["MAX_TIME"] = t_col 
       
        toolbox = base.Toolbox()
        toolbox.register("individual", init_ind)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", ga.crossover_un_punto)
        toolbox.register("mutate", ga.mutazione_gene)
        toolbox.register("select", tools.selNSGA2)
        pop = toolbox.population(n=POP_SIZE)


        logger.info(f"--- Experiment {exp+1}/{n_experiments} with entry {ing_early}, exit {usc_early}, speed {speed_early} ---")
        logger.info(f"collision_time: {t_col} min, exit_time_late: {distance(ing_late, usc_late)/speed_late} min \n ")
        logger.info(f"--- Starting population {exp+1}/{n_experiments} with entry {ing_late}, exit {usc_late}, speed {speed_late} ---")


        '''
        pop_time = [ind[0][0] for ind in pop]
        pop_theta = [ind[0][1] for ind in pop]
        pop_vel = [ind[0][2] for ind in pop]
        fig, ax = plt.subplots(3, 1)
        ax[0].hist(pop_time, bins=20, alpha=0.5, label='time')
        ax[1].hist(pop_theta, bins=20, alpha=0.5, label='theta')
        ax[2].hist(pop_vel, bins=20, alpha=0.5, label='velocity')
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[2].legend(loc='upper right')
        plt.show()
        sys.exit(0)
        '''
        
        # Inizializza fitness
        
        for ind in pop:
            logger.info("individual:" +str(ind))

            ind.fitness.values = toolbox.evaluate(ind, p0=ing_late, theta0=THETA_0, time0=TIME_0, lato=scn["area_size"], exit_ref=usc_late)
            logger.info("path:" +str(ind.sim["path"]))

            
            xx = [x[0] for x in ind.sim["path"]]
            yy = [y[1] for y in ind.sim["path"]]
            plt.plot(xx, yy, color='blue', alpha=0.1 )
            plt.scatter(xx[1], yy[1], color='red', alpha=0.1)
            
        plt.plot([ing_late[0], usc_late[0]], [ing_late[1], usc_late[1]], color='orange')
        plt.savefig("data/debugpop.png")
        plt.figure()
        sys.exit(0)
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
                ind.fitness.values = toolbox.evaluate(ind,p0=ing_late, theta0=THETA_0, time0=TIME_0, lato=scn["area_size"], exit_ref=usc_late)

            tools.emo.assignCrowdingDist(offspring)
            pop = toolbox.select(pop + offspring, POP_SIZE)

        # Salva Pareto front finale
        ga.save_pareto_front(tools.sortNondominated(pop, len(pop), True)[0], "pareto_front.json")

if __name__ == "__main__":
    main()
