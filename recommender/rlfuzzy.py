import json
from matplotlib.pylab import pareto
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from typing import List, Dict
import os
import glob
import matplotlib.pyplot as plt
import sys


class FuzzyDecisionMaker:
    def __init__(self):
        # PATH LENGTH: lunghezza della spezzata in area 40x40
        # Range realistico: 0 a ~113 (circa 2 volte la diagonale)
        # Normalizzato a 0-200 per il fuzzy system
        self.path_length = ctrl.Antecedent(np.arange(0, 201, 1), 'path_length')
        self.path_length['corto'] = fuzz.trapmf(self.path_length.universe, [0, 0, 50, 80])
        self.path_length['medio'] = fuzz.trimf(self.path_length.universe, [60, 100, 140])
        self.path_length['lungo'] = fuzz.trapmf(self.path_length.universe, [120, 160, 200, 200])
        
        # EXIT DISTANCE: distanza sul perimetro dell'area 40x40
        # Range reale: 0 a 160 (perimetro = 4*40)
        # Normalizzato a 0-160
        self.exit_distance = ctrl.Antecedent(np.arange(0, 161, 1), 'exit_distance')
        self.exit_distance['vicino'] = fuzz.trapmf(self.exit_distance.universe, [0, 0, 30, 60])
        self.exit_distance['medio'] = fuzz.trimf(self.exit_distance.universe, [50, 80, 110])
        self.exit_distance['lontano'] = fuzz.trapmf(self.exit_distance.universe, [100, 130, 160, 160])
        
        # EXIT DIRECTION: angolo in gradi (0-360)
        # 0° = perfettamente allineato, 180° = direzione opposta
        self.exit_direction = ctrl.Antecedent(np.arange(0, 361, 1), 'exit_direction')
        # Allineato: vicino a 0° o 360° (stesso significato circolare)
        self.exit_direction['allineato'] = fuzz.trapmf(self.exit_direction.universe, [0, 0, 30, 60])
        # Parziale: tra 60° e 120°
        self.exit_direction['parziale'] = fuzz.trimf(self.exit_direction.universe, [50, 90, 130])
        # Perpendicolare: circa 90° o 270°
        self.exit_direction['perpendicolare'] = fuzz.trimf(self.exit_direction.universe, [120, 180, 240])
        # Opposto: circa 180°
        self.exit_direction['opposto'] = fuzz.trimf(self.exit_direction.universe, [230, 270, 310])
        # Quasi allineato (vicino a 360°)
        self.exit_direction['quasi_allineato'] = fuzz.trapmf(self.exit_direction.universe, [300, 330, 360, 360])
        
        # PREFERENCE: preferenza della soluzione (output)
        self.preference = ctrl.Consequent(np.arange(0, 101, 1), 'preference')
        self.preference['molto_bassa'] = fuzz.trapmf(self.preference.universe, [0, 0, 15, 30])
        self.preference['bassa'] = fuzz.trimf(self.preference.universe, [20, 35, 50])
        self.preference['media'] = fuzz.trimf(self.preference.universe, [40, 55, 70])
        self.preference['alta'] = fuzz.trimf(self.preference.universe, [60, 75, 85])
        self.preference['molto_alta'] = fuzz.trapmf(self.preference.universe, [80, 90, 100, 100])
        
        # REGOLE FUZZY COMPLETE
        self.rules = [
            # Percorso corto
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['vicino'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['molto_alta']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['vicino'] & self.exit_direction['parziale'], 
                     self.preference['alta']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['vicino'] & self.exit_direction['perpendicolare'], 
                     self.preference['media']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['vicino'] & self.exit_direction['opposto'], 
                     self.preference['bassa']),
            
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['medio'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['alta']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['medio'] & self.exit_direction['parziale'], 
                     self.preference['media']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['medio'] & self.exit_direction['perpendicolare'], 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['medio'] & self.exit_direction['opposto'], 
                     self.preference['molto_bassa']),
            
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['lontano'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['media']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['lontano'] & self.exit_direction['parziale'], 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['corto'] & self.exit_distance['lontano'] & 
                     (self.exit_direction['perpendicolare'] | self.exit_direction['opposto']), 
                     self.preference['molto_bassa']),
            
            # Percorso medio
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['vicino'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['alta']),
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['vicino'] & self.exit_direction['parziale'], 
                     self.preference['media']),
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['vicino'] & self.exit_direction['perpendicolare'], 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['vicino'] & self.exit_direction['opposto'], 
                     self.preference['molto_bassa']),
            
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['medio'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['media']),
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['medio'] & self.exit_direction['parziale'], 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['medio'] & 
                     (self.exit_direction['perpendicolare'] | self.exit_direction['opposto']), 
                     self.preference['molto_bassa']),
            
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['lontano'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['medio'] & self.exit_distance['lontano'], 
                     self.preference['molto_bassa']),
            
            # Percorso lungo
            ctrl.Rule(self.path_length['lungo'] & self.exit_distance['vicino'] & 
                     (self.exit_direction['allineato'] | self.exit_direction['quasi_allineato']), 
                     self.preference['media']),
            ctrl.Rule(self.path_length['lungo'] & self.exit_distance['vicino'] & self.exit_direction['parziale'], 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['lungo'] & self.exit_distance['vicino'] & 
                     (self.exit_direction['perpendicolare'] | self.exit_direction['opposto']), 
                     self.preference['molto_bassa']),
            
            ctrl.Rule(self.path_length['lungo'] & self.exit_distance['medio'], 
                     self.preference['bassa']),
            ctrl.Rule(self.path_length['lungo'] & self.exit_distance['lontano'], 
                     self.preference['molto_bassa']),
        ]
        
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)
    
    def evaluate_solution(self, path_length, exit_distance, exit_direction):
        """
        Valuta una soluzione usando il sistema fuzzy
        
        Args:
            path_length: lunghezza della spezzata (verrà normalizzato a 0-200)
            exit_distance: distanza sul perimetro (0-160, nessuna normalizzazione)
            exit_direction: angolo in gradi (0-360, nessuna normalizzazione)
        """
        try:
            # Normalizza path_length proporzionalmente
            # Assumendo max realistico ~113, mappiamo a 0-200
            path_length_norm = np.clip(path_length * (200.0 / 113.0), 0, 200)
            exit_distance = float(np.clip(exit_distance, 0, 160))
            exit_direction = float(np.clip(exit_direction, 0, 360))
            
            # Crea un nuovo simulatore per ogni valutazione
            simulator = ctrl.ControlSystemSimulation(self.control_system)
            
            # Imposta gli input
            simulator.input['path_length'] = path_length_norm
            simulator.input['exit_distance'] = exit_distance
            simulator.input['exit_direction'] = exit_direction
            
            # Computa l'output
            simulator.compute()
            
            return simulator.output['preference']
            
        except KeyError as e:
            print(f"KeyError: {e}")
            print(f"Input: path={path_length:.2f}→{path_length_norm:.2f}, dist={exit_distance:.2f}, dir={exit_direction:.2f}°")
            return self._calculate_fallback_preference(path_length, exit_distance, exit_direction)
            
        except ValueError as e:
            print(f"ValueError (regole non attivate): {e}")
            print(f"Input: path={path_length:.2f}, dist={exit_distance:.2f}, dir={exit_direction:.2f}°")
            return self._calculate_fallback_preference(path_length, exit_distance, exit_direction)
            
        except Exception as e:
            print(f"Errore generico: {e}")
            print(f"Input: path={path_length:.2f}, dist={exit_distance:.2f}, dir={exit_direction:.2f}°")
            return 50.0
    
    def _calculate_fallback_preference(self, path_length, exit_distance, exit_direction):
        """Calcola una preferenza euristica quando il fuzzy system fallisce"""
        # Normalizza in [0, 1] dove 1 = migliore
        pl_norm = 1.0 - min(path_length / 113.0, 1.0)  # Più corto = meglio
        ed_norm = 1.0 - (exit_distance / 160.0)  # Più vicino = meglio
        
        # Per l'angolo, considera che 0° e 360° sono ottimali, 180° è peggiore
        angle_deviation = min(exit_direction, 360 - exit_direction)  # Distanza da 0°/360°
        edir_norm = 1.0 - (angle_deviation / 180.0)  # Più allineato = meglio
        
        # Media ponderata
        preference = (pl_norm * 0.35 + ed_norm * 0.35 + edir_norm * 0.30) * 100.0
        
        return np.clip(preference, 0, 100)
    
    def select_solution(self, pareto_solutions, epsilon=0.1):
        """Seleziona una soluzione dal fronte di Pareto"""
        if not pareto_solutions:
            return 0, np.array([50.0])
        
        preferences = []
        for sol in pareto_solutions:
            pref = self.evaluate_solution(
                sol['path_length'], 
                sol['exit_distance'], 
                sol['exit_direction']
            )
            preferences.append(pref)
        
        preferences = np.array(preferences)
        
        if np.random.random() < epsilon:
            if preferences.sum() > 0:
                probs = preferences / preferences.sum()
            else:
                probs = np.ones(len(preferences)) / len(preferences)
            selected_idx = np.random.choice(len(pareto_solutions), p=probs)
        else:
            selected_idx = np.argmax(preferences)
        
        return selected_idx, preferences
    
    def visualize(self):
        """Visualizza le funzioni di appartenenza"""
        os.makedirs('data', exist_ok=True)
        
        fig, axes = plt.subplots(nrows=4, figsize=(12, 14))
        
        # Path Length
        for label in self.path_length.terms:
            axes[0].plot(self.path_length.universe, self.path_length[label].mf, linewidth=2, label=label)
        axes[0].set_title('Path Length (normalizzato 0-200, reale ~0-113)')
        axes[0].set_ylabel('Grado di appartenenza')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Exit Distance
        for label in self.exit_distance.terms:
            axes[1].plot(self.exit_distance.universe, self.exit_distance[label].mf, linewidth=2, label=label)
        axes[1].set_title('Exit Distance sul perimetro 40x40 (0-160)')
        axes[1].set_ylabel('Grado di appartenenza')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Exit Direction
        for label in self.exit_direction.terms:
            axes[2].plot(self.exit_direction.universe, self.exit_direction[label].mf, linewidth=2, label=label)
        axes[2].set_title('Exit Direction in gradi (0-360°)')
        axes[2].set_ylabel('Grado di appartenenza')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Preference
        for label in self.preference.terms:
            axes[3].plot(self.preference.universe, self.preference[label].mf, linewidth=2, label=label)
        axes[3].set_title('Preference - Output (0-100)')
        axes[3].set_ylabel('Grado di appartenenza')
        axes[3].set_xlabel('Valore')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/membership_functions.png', dpi=150)
        plt.close(fig)
        print("Grafico salvato in 'data/membership_functions.png'")

class GenotypeEncoder:
    def __init__(self, max_waypoints=10, waypoint_dim=3):
        self.max_waypoints = max_waypoints
        self.waypoint_dim = waypoint_dim
        self.encoding_dim = max_waypoints * waypoint_dim
    
    def encode_genotype(self, waypoints):
        encoded = np.zeros(self.encoding_dim, dtype=np.float32)
        waypoints = waypoints[:self.max_waypoints]
        for i, wp in enumerate(waypoints):
            start_idx = i * self.waypoint_dim
            end_idx = start_idx + min(len(wp), self.waypoint_dim)
            encoded[start_idx:end_idx] = wp[:self.waypoint_dim]
        return encoded
    
    def encode_genotype_stats(self, waypoints):
        if not waypoints:
            return np.zeros(8, dtype=np.float32)
        wp_array = np.array(waypoints)
        num_wp = len(waypoints)
        avg_x = np.mean(wp_array[:,0])
        avg_y = np.mean(wp_array[:,1])
        std_x = np.std(wp_array[:,0])
        std_y = np.std(wp_array[:,1])
        distances = np.sqrt(np.sum(np.diff(wp_array[:, :2], axis=0)**2, axis=1))
        total_distance = np.sum(distances)
        max_dev_x = np.max(np.abs(wp_array[:,0] - wp_array[0,0]))
        max_dev_y = np.max(np.abs(wp_array[:,1] - wp_array[0,1]))
        return np.array([num_wp, avg_x, avg_y, std_x, std_y, total_distance, max_dev_x, max_dev_y], dtype=np.float32)

class MultiScenarioEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, input_folder: str, fuzzy_decider: FuzzyDecisionMaker,
                 epsilon=0.1, use_genotype_encoding='stats', max_waypoints=10):
        super().__init__()
        self.scenarios = []
        file_list = glob.glob(os.path.join(input_folder, "*_pareto_front.json"))
        
        if not file_list:
            raise ValueError(f"Nessun file trovato in {input_folder}")
        
        for idx, filename in enumerate(file_list):
            try:
                with open(filename) as f:
                    d = json.load(f)
                    scenario = d.get("scenario", {})
                    population = d.get("population", [])
                    
                    if not population or not scenario:
                        continue
                    
                    # Raccogli tutti i fitness per calcolare min/max
                    all_fitness = []
                    for p in population:
                        fitness = p.get("fitness", [])
                        if len(fitness) >= 3:
                            all_fitness.append(fitness[:3])
                    
                    if not all_fitness:
                        continue
                    
                    all_fitness = np.array(all_fitness)
                    
                    # Calcola min e max per ogni obiettivo
                    min_vals = all_fitness.min(axis=0)
                    max_vals = all_fitness.max(axis=0)
                    
                    pareto_front = []
                    genotypes = []
                    
                    for p in population:
                        fitness = p.get("fitness", [])
                        genome = p.get("genome", [])
                        
                        if len(fitness) < 3 or not genome:
                            continue
                        
                        # Normalizza nel range del fuzzy system
                        # path_length -> deviation (0-200)
                        # exit_distance -> fuel (0-50)  
                        # exit_direction -> time (0-120)
                        deviation_norm = self._normalize(
                            fitness[0], min_vals[0], max_vals[0], 0, 200
                        )
                        fuel_norm = self._normalize(
                            fitness[1], min_vals[1], max_vals[1], 0, 50
                        )
                        time_norm = self._normalize(
                            fitness[2], min_vals[2], max_vals[2], 0, 120
                        )
                        
                        pareto_front.append({
                            "path_length": fitness[0],  # Mantieni il valore reale (0-~113)
                            "exit_distance": fitness[1],  # Mantieni il valore reale (0-160)
                            "exit_direction": fitness[2],  # Mantieni il valore reale (0-360)
                            # Salva anche i valori originali
                            "path_length_original": fitness[0],
                            "exit_distance_original": fitness[1],
                            "exit_direction_original": fitness[2]
                        })
                        genotypes.append(genome)
                    
                    if not pareto_front:
                        continue
                    
                    scenario_struct = {
                        "scenario_id": idx,
                        "collision_time": scenario.get("collision_time", 0),
                        "collision_distance": scenario.get("area_size", 0),
                        "relative_velocity": scenario.get("speed_late", 0),
                        "pareto_front": pareto_front,
                        "genotypes": genotypes,
                        # Salva i range per riferimento
                        "fitness_ranges": {
                            "path_length_min": min_vals[0],
                            "path_length_max": max_vals[0],
                            "exit_distance_min": min_vals[1],
                            "exit_distance_max": max_vals[1],
                            "exit_direction_min": min_vals[2],
                            "exit_direction_max": max_vals[2]
                        }
                    }
                    self.scenarios.append(scenario_struct)
            
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Errore nel caricamento di {filename}: {e}")
                continue
        
        if not self.scenarios:
            raise ValueError("Nessuno scenario valido caricato")
        
        self.fuzzy_decider = fuzzy_decider
        self.epsilon = epsilon
        self.genotype_encoder = GenotypeEncoder(max_waypoints=max_waypoints)
        self.use_genotype_encoding = use_genotype_encoding
        
        if use_genotype_encoding == 'stats':
            genotype_obs_dim = 8
        elif use_genotype_encoding == 'flat':
            genotype_obs_dim = self.genotype_encoder.encoding_dim
        else:
            genotype_obs_dim = 0
        
        obs_dim = 7 + genotype_obs_dim
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(100)
        self.current_scenario = None
    
    def _normalize(self, value, old_min, old_max, new_min, new_max):
        """
        Normalizza un valore da un range ad un altro
        
        Args:
            value: valore da normalizzare
            old_min: minimo del range originale
            old_max: massimo del range originale
            new_min: minimo del nuovo range
            new_max: massimo del nuovo range
        
        Returns:
            valore normalizzato
        """
        if old_max - old_min == 0:
            # Se tutti i valori sono uguali, ritorna il punto medio del nuovo range
            return (new_min + new_max) / 2.0
        
        # Formula di normalizzazione lineare
        normalized = new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)
        
        # Clipping per sicurezza
        return np.clip(normalized, new_min, new_max)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_scenario = self.scenarios[self.np_random.integers(len(self.scenarios))]
        observation = self._get_observation()
        info = {
            'scenario_id': self.current_scenario['scenario_id'],
            'fitness_ranges': self.current_scenario['fitness_ranges']
        }
        
        return observation, info
    
    def _get_observation(self, selected_pareto_idx=None):
        scenario = self.current_scenario
        pareto = scenario['pareto_front']
        
        avg_path_length = np.mean([s['path_length'] for s in pareto])
        avg_exit_distance = np.mean([s['exit_distance'] for s in pareto])
        avg_exit_direction = np.mean([s['exit_direction'] for s in pareto])

        base_obs = np.array([
            scenario['collision_time'],
            scenario['collision_distance'],
            scenario['relative_velocity'],
            avg_path_length,
            avg_exit_distance,
            avg_exit_direction,
            len(pareto)
        ], dtype=np.float32)
                
     
        
        if 'genotypes' in scenario and selected_pareto_idx is not None:
            genotype = scenario['genotypes'][selected_pareto_idx]
            if self.use_genotype_encoding == 'stats':
                genotype_features = self.genotype_encoder.encode_genotype_stats(genotype)
            elif self.use_genotype_encoding == 'flat':
                genotype_features = self.genotype_encoder.encode_genotype(genotype)
            else:
                genotype_features = np.array([])
        else:
            if 'genotypes' in scenario and len(scenario['genotypes']) > 0:
                all_stats = [self.genotype_encoder.encode_genotype_stats(g) for g in scenario['genotypes']]
                genotype_features = np.mean(all_stats, axis=0)
            else:
                if self.use_genotype_encoding == 'stats':
                    genotype_features = np.zeros(8, dtype=np.float32)
                elif self.use_genotype_encoding == 'flat':
                    genotype_features = np.zeros(self.genotype_encoder.encoding_dim, dtype=np.float32)
                else:
                    genotype_features = np.array([])
        
        if len(genotype_features) > 0:
            obs = np.concatenate([base_obs, genotype_features])
        else:
            obs = base_obs
        
        return obs
    
    def step(self, action):
        pareto = self.current_scenario['pareto_front']
        action = int(action) % len(pareto)
        
        # Il fuzzy decider usa i valori normalizzati
        selected_idx, all_preferences = self.fuzzy_decider.select_solution(pareto, epsilon=self.epsilon)
        
        suggested_solution = pareto[action]
        chosen_solution = pareto[selected_idx]
        
        if action == selected_idx:
            reward = 10.0
        else:
            # Calcola distanza usando i valori normalizzati
            dist = np.sqrt(
                ((suggested_solution['path_length'] - chosen_solution['path_length']) / 113.0) ** 2 +
                ((suggested_solution['exit_distance'] - chosen_solution['exit_distance']) / 160.0) ** 2 +
                ((suggested_solution['exit_direction'] - chosen_solution['exit_direction']) / 360.0) ** 2
            )
            pref_diff = abs(all_preferences[action] - all_preferences[selected_idx]) / 100.0
            reward = -5.0 * dist - 3.0 * pref_diff
        
        quality_bonus = (all_preferences[action] / 100.0) * 2.0
        reward += quality_bonus
        
        if 'genotypes' in self.current_scenario:
            genotype_similarity = self._compute_genotype_similarity(
                self.current_scenario['genotypes'][action],
                self.current_scenario['genotypes'][selected_idx]
            )
            reward += genotype_similarity * 1.5
        
        terminated, truncated = True, False
        
        info = {
            'suggested_idx': action,
            'chosen_idx': selected_idx,
            'match': action == selected_idx,
            'suggested_preference': all_preferences[action],
            'chosen_preference': all_preferences[selected_idx],
            'all_preferences': all_preferences.tolist(),
            # Aggiungi valori originali per debugging
            'suggested_original': {
                'path_length': suggested_solution.get('path_length_original'),
                'exit_distance': suggested_solution.get('exit_distance_original'),
                'exit_direction': suggested_solution.get('exit_direction_original')
            },
            'chosen_original': {
                'path_length': chosen_solution.get('path_length_original'),
                'exit_distance': chosen_solution.get('exit_distance_original'),
                'exit_direction': chosen_solution.get('exit_direction_original')
            }
        }
        
        observation = self._get_observation()
        return observation, reward, terminated, truncated, info
    
    def _compute_genotype_similarity(self, geno1, geno2):
        if not geno1 or not geno2:
            return 0.0
        g1 = np.array(geno1)
        g2 = np.array(geno2)
        min_len = min(len(g1), len(g2))
        g1 = g1[:min_len]
        g2 = g2[:min_len]
        dist = np.sqrt(np.sum((g1 - g2) ** 2))
        max_dist = np.sqrt(np.sum(g1 ** 2) + np.sum(g2 ** 2))
        if max_dist > 0:
            similarity = 1.0 - (dist / max_dist)
        else:
            similarity = 1.0
        return np.clip(similarity, 0.0, 1.0)

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_accuracies = []
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'match' in info:
                self.episode_accuracies.append(1.0 if info['match'] else 0.0)
        return True
    def _on_rollout_end(self):
        if len(self.episode_accuracies) > 0 and self.verbose > 0:
            recent_acc = np.mean(self.episode_accuracies[-100:])
            print(f"Accuracy recente (ultimi 100): {recent_acc:.3f}")

def train_from_folder(input_folder="./", total_timesteps=100000, encoding_type='stats'):
    fuzzy_decider = FuzzyDecisionMaker()
    fuzzy_decider.visualize()
    
    env = MultiScenarioEnv(input_folder, fuzzy_decider, epsilon=0.15, use_genotype_encoding=encoding_type)
    
    # CORREZIONE: formato corretto per net_arch
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        batch_size=64, 
        n_epochs=10, 
        n_steps=2048, 
        device='cpu',
        policy_kwargs=dict(net_arch=dict(pi=[256,256], vf=[256,256]))  # Dizionario invece di lista
    )
    
    callback = MetricsCallback(verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("pareto_recommender_multi_scenario_model")
    print("Addestramento concluso. Modello salvato in 'pareto_recommender_multi_scenario_model'.")


if __name__ == "__main__":
    train_from_folder(input_folder="./data", total_timesteps=100000, encoding_type='stats')