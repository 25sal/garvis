import csv
import random
import math
import argparse
import matplotlib.pyplot as plt

# Definizione delle dimensioni del piano (in miglia nautiche)
area_size = 40  # Piano 40 x 40
max_distance = 2.5  # Raggio massimo per l'incontro
speed_min = 500  # Velocità minima (km/h)
speed_max = 800  # Velocità massima (km/h)
n_experiments = 10


# Funzione per generare un punto casuale all'interno del piano
def generate_random_point(size):
    return random.uniform(0, size), random.uniform(0, size)

# Funzione per calcolare l'intersezione con il bordo dell'area
def calculate_border_intersection(x, y, angle, size):
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # Troviamo i punti di intersezione con i bordi
    t_x_min = -x / dx if dx != 0 else float('inf')
    t_x_max = (size - x) / dx if dx != 0 else float('inf')
    t_y_min = -y / dy if dy != 0 else float('inf')
    t_y_max = (size - y) / dy if dy != 0 else float('inf')
    
    # Calcoliamo i tempi di uscita più piccoli e positivi
    t_enter = max(min(t_x_min, t_x_max), min(t_y_min, t_y_max))
    t_exit = min(max(t_x_min, t_x_max), max(t_y_min, t_y_max))
    
    # Punti di ingresso e uscita
    x_in, y_in = x + t_enter * dx, y + t_enter * dy
    x_out, y_out = x + t_exit * dx, y + t_exit * dy
    
    return (x_in, y_in), (x_out, y_out)

# Funzione per leggere i dati dal file CSV
def leggi_dati_csv(file_path):
    """
    Legge i dati degli aerei da un file CSV e li restituisce come lista.

    :param file_path: Percorso al file CSV.
    :return: Lista con i dati degli aerei e il punto di incontro.
    """
    aerei_data = []
    collision_points = []
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            ingresso = (float(row["Ingresso_x"]), float(row["Ingresso_y"]))
            uscita = (float(row["Uscita_x"]), float(row["Uscita_y"]))
            p_incontro = (float(row["PuntoIncontro_x"]), float(row["PuntoIncontro_y"]))
            velocita = float(row["Velocità_kmh"])
            aerei_data.append((ingresso, uscita, p_incontro, velocita))
            collision_points.append(p_incontro)
    return aerei_data, collision_points

# Funzione per visualizzare le traiettorie
def visualizza_traiettorie(aerei_data, area_size, p_incontro, print=False):
    """
    Visualizza le traiettorie degli aerei su un piano con Matplotlib.

    :param aerei_data: Lista contenente i dati degli aerei (ingresso, uscita, velocità).
    :param area_size: Dimensione del piano (lato del quadrato).
    :param p_incontro: Coordinate del punto di incontro.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

   
    
    for i in range(0,len(aerei_data),2):
        # Disegna il bordo dell'area
        ax.plot([0, area_size, area_size, 0, 0], [0, 0, area_size, area_size, 0], 'k-', lw=2, label="Bordo area")
        # Imposta i limiti dell'area
        ax.set_xlim(0, area_size)
        ax.set_ylim(0, area_size)
        ax.set_aspect('equal', adjustable='box')

        # Aggiungi titolo, legenda e griglia
        ax.set_title("Traiettorie degli aerei")
        ax.set_xlabel("Miglia nautiche (x)")
        ax.set_ylabel("Miglia nautiche (y)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        # Aggiungi il punto di incontro
        ax.scatter(*p_incontro[i], color="red", s=100, label="Punto di incontro")

        # Disegna le traiettorie degli aerei
        for j, aereo in enumerate(aerei_data[i:i+2]):
            ingresso, uscita, _, speed = aereo
            distance = ((ingresso[0] - p_incontro[i][0])**2 + (ingresso[1] - p_incontro[i][1])**2)**0.5
            time_flight = distance / speed
            ax.plot([ingresso[0], uscita[0]], [ingresso[1], uscita[1]], label=f"Aereo {int(i/2)}_{j} (Vel: {speed:.1f} km/h) t={time_flight:.2f}h")
            ax.scatter(*ingresso, color="blue", s=50, label=f"Ingresso Aereo {int(i/2)}_{j}" if i == 0 else None)
            ax.scatter(*uscita, color="green", s=50, label=f"Uscita Aereo {int(i/2)}_{j}" if i == 0 else None)
        if print:
            plt.savefig(f"data/2dscenario_{int(i/2)}.png")
            #rimuovi le traiettorie e il punto di incontro per la prossima iterazione
            ax.cla()

    if not print:
        # Mostra il grafico
        plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generatore di traiettorie aeree 2D")
    parser.add_argument("command", type=str, choices=["generate", "visualize"], help="Comando da eseguire")
    # add parameters for generation
    parser.add_argument("--output", type=str, default="data/traiettorie_aerei.csv", help="File di output per le traiettorie")
    parser.add_argument("--n_experiments", type=int, default=10, help="Numero di esperimenti da generare")
    parser.add_argument("--area_size", type=float, default=40.0, help="Dimensione del piano (lato del quadrato)")
    parser.add_argument("--speed_min", type=float, default=500.0, help="Velocità minima degli aerei (km/h)")
    parser.add_argument("--speed_max", type=float, default=800.0, help="Velocità massima degli aerei (km/h)")   
    # add parameters for visualization
    parser.add_argument("--input", type=str, default="data/traiettorie_aerei.csv", help="File di input per la visualizzazione delle traiettorie")
    #number of trajectories to show
    parser.add_argument("--av", type=int, default=6, help="Numero di traiettorie da visualizzare")
    parser.add_argument("--print", action="store_true", help="Stampa i dati delle traiettorie")
    
    args = parser.parse_args()
    if args.command == "generate":
        n_experiments = args.n_experiments
        area_size = args.area_size
        speed_min = args.speed_min
        speed_max = args.speed_max
        output_file = args.output
        aerei_data = []

        for _ in range(n_experiments):
            # Generazione dei dati per due aerei
            p_incontro = generate_random_point(area_size)
            for _ in range(2):
                angle = random.uniform(0, 2 * math.pi)  # Direzione casuale
                speed = random.uniform(speed_min, speed_max)  # Velocità casuale
                ingresso, uscita = calculate_border_intersection(p_incontro[0], p_incontro[1], angle, area_size)
                aerei_data.append((ingresso, uscita, p_incontro, speed))

        # Salvataggio in un file CSV
        output_file = "data/traiettorie_aerei.csv"
        id  = 0
        experiment = 0
        with open(output_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Ingresso_x", "Ingresso_y", "Uscita_x", "Uscita_y", "PuntoIncontro_x", "PuntoIncontro_y", "Velocità_kmh"])
            
            for aereo in aerei_data:
                ingresso, uscita, p_incontro, speed = aereo
                writer.writerow([
                    ingresso[0], ingresso[1],
                    uscita[0], uscita[1],
                    p_incontro[0], p_incontro[1],
                    speed, str(experiment)+"_"+str(id)
                ])
                id += 1%2
            experiment += 1

        print(f"File CSV generato: {output_file}")
    elif args.command == "visualize":
        trajs_to_show = args.av
        # Visualizzazione delle traiettorie da un file CSV
        input_file = args.input
        aerei_data, collision_points = leggi_dati_csv(input_file)
        if len(aerei_data) > trajs_to_show:
            aerei_data = aerei_data[:trajs_to_show]      
        visualizza_traiettorie(aerei_data, area_size, collision_points, print=args.print)
        