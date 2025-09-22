import matplotlib.pyplot as plt
import json
import logging
from shapely.geometry import LineString, Polygon

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

exp = 0
filename = f"data/pareto_scenario_{0}.json"
results = json.load(open(filename))
scenario = results['scenario']
seed = results['seed']
in_early = scenario['early']['ingress']
usc_early = scenario['early']['exit']
speed_early = scenario['early']['speed']
in_late = scenario['late']['ingress']
usc_late = scenario['late']['exit']
traj_early = LineString([in_early, usc_early])
traj_late = LineString([in_late, usc_late])
p_inc = scenario['collision_point']
area_size = scenario['area_size']
# Visualizzazione "/data/didattica/tesisti/specialistica/2025_disiero/codice/final/ga/disiero.py", line 128dei risultati
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(*traj_early.xy, color='red', linewidth=2, label='A1')
ax.plot(*traj_late.xy, color='purple', linewidth=2, label='A2 original')
ax.plot(*in_early, 'o', color='blue')
ax.plot(*usc_early, 'o', color='green')
ax.plot(*in_late, 'o', color='blue')
ax.plot(*usc_late, 'o', color='green')
ax.plot(p_inc[0], p_inc[1], 'x', color='black', markersize=10, label='Collision Point')
# Aggiunta alla legenda del significato dei colori
ax.plot([], [], 'o', color='blue', label='Punto di ingresso')
ax.plot([], [], 'o', color='green', label='Punto di uscita')

pareto_front = results['pareto_front']

'''
def is_identical(traj1, traj2, tolerance=1e-6):
    return all(distance(p1, p2) < tolerance for p1, p2 in zip(traj1, traj2))


print(f" Confronto con seed:")
for i, traj in enumerate(pareto_front):
    try:
        if is_identical(traj, seed):
            logger.info(f" Traj {i+1} è identica alla seed.")
        else:
            logger.info(f" Traj {i+1} è diversa dalla seed.")
    except Exception as e:
        logger.info(f"Errore nel confronto per la traiettoria {i+1}: {e}")


#  Metodo 2 — stampa le coordinate delle prime 3 traiettorie evolute
for i, traj in enumerate(pareto_front[:3]):
    print(f"\n[EVOLUTO] Traj {i+1} - Coordinate:")
    for point in traj:
        print(f"  {point}")



'''

# Disegna la seed in rosso tratteggiato
x_seed, y_seed = zip(*seed)
ax.plot(x_seed, y_seed, linestyle='--', linewidth=2, color='black', label='Seed')
# Per maggiore chiarezza, porta in primo piano le leggende
ax.legend()

ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)
ax.set_aspect('equal')
ax.set_title("NSGA-II - Traiettorie alternative")
ax.set_xlabel("NM (x)")
ax.set_ylabel("NM (y)")
ax.grid(True, linestyle='--', alpha=0.5)


for ind, traj in enumerate(pareto_front):


    ax.plot(*zip(*traj['path']), linewidth=1, label=f'Path {ind+1}')

ax.legend()
print("\nLegenda colori:")
print("Punto di ingresso (entry point) → blu")
print("Punto di uscita (exit point) → verde")
plt.savefig(f"data/pareto_scenario_{exp}.png")

fig, ax = plt.subplots(figsize=(8, 8))
for ind, traj in enumerate(pareto_front):
    ax.scatter(traj['fitness'][0], traj['fitness'][1])
ax.set_xlabel("Funzione obiettivo 1: Lunghezza (NM)")
ax.set_ylabel("Funzione obiettivo 2: Distanza minima da A1 (NM)")
plt.savefig(f"data/pareto_front0_{exp}.png")

fig, ax = plt.subplots(figsize=(8, 8))
for ind, traj in enumerate(pareto_front):
    ax.scatter(traj['fitness'][1], traj['fitness'][2])
ax.set_xlabel("Funzione obiettivo 2: fluidità (NM)")
ax.set_ylabel("Funzione obiettivo 3: deviazione ")
plt.savefig(f"data/pareto_front1_{exp}.png")

fig, ax = plt.subplots(figsize=(8, 8))
for ind, traj in enumerate(pareto_front):
    ax.scatter(traj['fitness'][0], traj['fitness'][2])
ax.set_xlabel("Funzione obiettivo 1: Lunghezza (NM)")
ax.set_ylabel("Funzione obiettivo 3: Deviazione (NM)")
plt.savefig(f"data/pareto_front2_{exp}.png")
    