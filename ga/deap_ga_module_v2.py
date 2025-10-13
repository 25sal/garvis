
import random
import math
import json
from typing import List, Tuple
from scenario.conf import scn
from shapely.geometry import LineString, Polygon, Point
from myutils.geometry2d import calculate_border_intersection, compute_initial_bearing
import logging
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Parametri globali
MAX_GENI = scn["n_segments"]
LATO_AREA = scn["area_size"]
SPEED_LIM = (scn["speed_min"], scn["speed_max"])
THETA_LIM = scn["max_turn_angle_deg"]






Gene = Tuple[float, float, float]  # (t, delta_theta, v)
Individuo = List[Gene]

def clamp(x, a, b):
    return max(a, min(b, x))

def ordina_individuo(ind: Individuo) -> Individuo:
    return sorted(ind, key=lambda g: g[0])

def simula_traiettoria(individuo: Individuo,
                       p0,
                       theta0,
                       t0,
                       lato):
    x, y = p0
    theta = theta0
    
    t_prev = t0
    path = [(x, y, t_prev)]
    exited = False
    time_of_exit = None
    exit_point = None
    path_length_inside = 0.0
    total_angle_change = 0.0
    border = Polygon([(0, 0), (0, lato), (lato, lato), (lato, 0)])
    border = border.boundary
    
    for t, dtheta, v in ordina_individuo(individuo):
        dt = max(0.0, t - t_prev)
        dx = v * math.cos(math.radians(theta)) * dt
        dy = v * math.sin(math.radians(theta)) * dt
        new_x = x + dx
        new_y = y + dy
        new_t = t
        line = LineString([(x, y), (new_x, new_y)])
        
        # compute the intersection point between the line and the border using shapely  
        inter = line.intersection(border)
        #if the intersection is not empty
        if inter is not None and  inter.distance(Point(p0))>0:
            new_x, new_y = inter.x, inter.y
            line = LineString([(x, y), (new_x, new_y)])
            new_t = t_prev + line.length / v
            exited = True
        
        seg_len = line.length
        path_length_inside += seg_len
        path.append((new_x, new_y, new_t))       
        x, y, t_prev = new_x, new_y, new_t
        theta = theta + dtheta
        total_angle_change += abs(dtheta)

        if exited:
            time_of_exit = new_t
            exit_point = (x, y)
            break

   

    if not exited:
        
        # extend the last line segment to the border  
        dx = 2*lato*math.cos(math.radians(theta))
        dy = 2*lato*math.sin(math.radians(theta))
        line = LineString([(path[-1][0], path[-1][1]), (path[-1][0]+dx, path[-1][1]+dy)])
        inter = line.intersection(border)
        line = LineString([(path[-1][0], path[-1][1]), (inter.x, inter.y)])
        path_length_inside += line.length
        time_of_exit = path[-1][2] + line.length / v
        path.append((inter.x, inter.y, time_of_exit))
        exit_point = (inter.x,inter.y)
        exited = True

 
 
            
    # print(path)
    return {
        "path": path,
        "exited": exited,
        "time_of_exit": time_of_exit,
        "exit_point": exit_point,
        "path_length_inside": path_length_inside,
        "total_angle_change": total_angle_change
    }

def valuta_individuo(individuo: Individuo,
                     p0,
                     theta0,
                     t0,
                     lato,
                     exit_ref):
    sim = simula_traiettoria(individuo, p0=p0, theta0=theta0, t0=t0, lato=lato)
    length = sim["path_length_inside"]
    if sim["exited"] and sim["exit_point"] is not None:
        ex, ey = sim["exit_point"]
        dist_exit = math.hypot(ex - exit_ref[0], ey - exit_ref[1])
    else:
        dist_exit = math.hypot(lato/2 - exit_ref[0], lato/2 - exit_ref[1])
    fluidity = sim["total_angle_change"]
    return (length, dist_exit, fluidity), sim

def random_gene(time_span, speed_lim, max_dtheta=90.0):
    t = random.uniform(time_span[0], time_span[1])
    dtheta = random.uniform(-max_dtheta, max_dtheta)
    v = random.uniform(speed_lim[0], speed_lim[1])
    return (t, dtheta, v)

def init_individuo_random(num_geni: int, speed_lim, theta_lim, time_span):
    #num_geni = int(clamp(num_geni, 0, MAX_GENI))
    '''
    probabilmente generando in ordine (con time span decrescente, si ottimizzano le performance)
    '''
    ind = [random_gene(time_span, speed_lim=speed_lim, max_dtheta=theta_lim) for _ in range(num_geni)]
    return ordina_individuo(ind)

def mutazione_gene(individuo: Individuo,
                   sigma_t=0.5, sigma_theta=10.0, sigma_v=0.5,  max_v=5.0):
   
    max_time=scn["MAX_TIME"]
    nuovo = individuo.copy()
    if not nuovo:
        return nuovo
    idx = random.randrange(len(nuovo))
    t, dtheta, v = nuovo[idx]
    scelta = random.choice(["tempo", "theta", "vel"])
    if scelta == "tempo":
        t = clamp(t + random.gauss(0, sigma_t), 0.1, max_time)
    elif scelta == "theta":
        dtheta = (dtheta + random.gauss(0, sigma_theta)) % 360.0
    else:
        v = max(0.1, v + random.gauss(0, sigma_v))
    nuovo[idx] = (t, dtheta, v)
    return ordina_individuo(nuovo)

def inserimento_gene(individuo: Individuo, max_v=5.0):
    max_time=scn["MAX_TIME"]
    nuovo = individuo.copy()
    if len(nuovo) < MAX_GENI:
        nuovo.append(random_gene(max_time=max_time, max_v=max_v))
    return ordina_individuo(nuovo)

def elimina_gene(individuo: Individuo):
    nuovo = individuo.copy()
    if nuovo:
        idx = random.randrange(len(nuovo))
        del nuovo[idx]
    return ordina_individuo(nuovo)

def crossover_un_punto(p1: Individuo, p2: Individuo):
    if len(p1) < 1 or len(p2) < 1:
        return p1.copy(), p2.copy()
    cut1 = random.randint(1, len(p1))
    cut2 = random.randint(1, len(p2))
    child1 = ordina_individuo(p1[:cut1] + p2[cut2:])
    child2 = ordina_individuo(p2[:cut2] + p1[cut1:])
    return child1[:MAX_GENI], child2[:MAX_GENI]

def save_pareto_front(population, filename="pareto_front.json"):
    data = []
    for ind in population:
        fitness = getattr(ind, "fitness", None)
        fit_vals = tuple(getattr(fitness, "values", ())) if fitness else ()
        sim = getattr(ind, "sim", None)
        entry = {
            "genome": ind,
            "fitness": fit_vals,
        }
        if sim:
            entry.update({
                "path": sim.get("path"),
                "exited": sim.get("exited"),
                "time_of_exit": sim.get("time_of_exit"),
                "exit_point": sim.get("exit_point")
            })
        data.append(entry)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return filename
