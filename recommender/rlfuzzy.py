import json
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

class FuzzyDecisionMaker:
    def __init__(self):
        self.deviation = ctrl.Antecedent(np.arange(0, 201, 1), 'deviation')
        self.deviation['bassa'] = fuzz.trapmf(self.deviation.universe, [0, 0, 30, 60])
        self.deviation['media'] = fuzz.trimf(self.deviation.universe, [40, 80, 120])
        self.deviation['alta'] = fuzz.trapmf(self.deviation.universe, [100, 150, 200, 200])
        self.fuel = ctrl.Antecedent(np.arange(0, 51, 1), 'fuel')
        self.fuel['basso'] = fuzz.trapmf(self.fuel.universe, [0, 0, 5, 15])
        self.fuel['medio'] = fuzz.trimf(self.fuel.universe, [10, 20, 30])
        self.fuel['alto'] = fuzz.trapmf(self.fuel.universe, [25, 40, 50, 50])
        self.time = ctrl.Antecedent(np.arange(0, 121, 1), 'time')
        self.time['rapido'] = fuzz.trapmf(self.time.universe, [0, 0, 20, 40])
        self.time['moderato'] = fuzz.trimf(self.time.universe, [30, 60, 90])
        self.time['lento'] = fuzz.trapmf(self.time.universe, [80, 100, 120, 120])
        self.preference = ctrl.Consequent(np.arange(0, 101, 1), 'preference')
        self.preference['molto_bassa'] = fuzz.trapmf(self.preference.universe, [0, 0, 15, 30])
        self.preference['bassa'] = fuzz.trimf(self.preference.universe, [20, 35, 50])
        self.preference['media'] = fuzz.trimf(self.preference.universe, [40, 55, 70])
        self.preference['alta'] = fuzz.trimf(self.preference.universe, [60, 75, 85])
        self.preference['molto_alta'] = fuzz.trapmf(self.preference.universe, [80, 90, 100, 100])
        self.rules = [
            ctrl.Rule(self.deviation['bassa'] & self.fuel['basso'] & self.time['rapido'], self.preference['molto_alta']),
            ctrl.Rule(self.deviation['bassa'] & self.fuel['basso'], self.preference['alta']),
            ctrl.Rule(self.deviation['bassa'] & self.fuel['medio'] & self.time['rapido'], self.preference['alta']),
            ctrl.Rule(self.deviation['media'] & self.fuel['basso'] & self.time['rapido'], self.preference['alta']),
            ctrl.Rule(self.deviation['bassa'] & self.fuel['medio'], self.preference['media']),
            ctrl.Rule(self.deviation['media'] & self.fuel['medio'] & self.time['rapido'], self.preference['media']),
            ctrl.Rule(self.deviation['media'] & self.fuel['basso'], self.preference['media']),
            ctrl.Rule(self.deviation['bassa'] & self.fuel['alto'], self.preference['bassa']),
            ctrl.Rule(self.deviation['media'] & self.fuel['medio'], self.preference['bassa']),
            ctrl.Rule(self.deviation['alta'] & self.fuel['alto'], self.preference['molto_bassa']),
            ctrl.Rule(self.deviation['alta'] & self.fuel['medio'] & self.time['lento'], self.preference['molto_bassa']),
            ctrl.Rule(self.deviation['alta'], self.preference['bassa']),
            ctrl.Rule(self.time['lento'] & self.fuel['alto'], self.preference['bassa']),
            ctrl.Rule(self.time['rapido'] & self.deviation['media'] & self.fuel['medio'], self.preference['media']),
        ]
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)
    
    def evaluate_solution(self, deviation, fuel, time):
        deviation = np.clip(deviation, 0, 200)
        fuel = np.clip(fuel, 0, 50)
        time = np.clip(time, 0, 120)
        self.simulator.input['deviation'] = deviation
        self.simulator.input['fuel'] = fuel
        self.simulator.input['time'] = time
        self.simulator.compute()
        return self.simulator.output['preference']

    def select_solution(self, pareto_solutions, epsilon=0.1):
        if not pareto_solutions:
            return 0, np.array([50.0])
        preferences = np.array([self.evaluate_solution(sol['deviation'], sol['fuel'], sol['time']) for sol in pareto_solutions])
        if np.random.random() < epsilon:
            probs = preferences / preferences.sum()
            selected_idx = np.random.choice(len(pareto_solutions), p=probs)
        else:
            selected_idx = np.argmax(preferences)
        return selected_idx, preferences

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
        file_list = glob.glob(os.path.join(input_folder, "*.json"))
        for idx, filename in enumerate(file_list):
            with open(filename) as f:
                d = json.load(f)
                scenario = d.get("scenario", {})
                population = d.get("population", [])
                if not population or not scenario:
                    continue
                pareto_front = []
                genotypes = []
                for p in population:
                    fitness = p.get("fitness", [])
                    genome = p.get("genome", [])
                    if len(fitness) < 3 or not genome:
                        continue
                    pareto_front.append({
                        "deviation": fitness[0],
                        "fuel": fitness[1],
                        "time": fitness[2]
                    })
                    genotypes.append(genome)
                scenario_struct = {
                    "scenario_id": idx,
                    "collision_time": scenario.get("collision_time", 0),
                    "collision_distance": scenario.get("area_size", 0),
                    "relative_velocity": scenario.get("speed_late", 0),
                    "pareto_front": pareto_front,
                    "genotypes": genotypes
                }
                self.scenarios.append(scenario_struct)
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
    
    def _get_observation(self, selected_pareto_idx=None):
        scenario = self.current_scenario
        pareto = scenario['pareto_front']
        avg_deviation = np.mean([s['deviation'] for s in pareto])
        avg_fuel = np.mean([s['fuel'] for s in pareto])
        avg_time = np.mean([s['time'] for s in pareto])
        base_obs = np.array([
            scenario['collision_time'],
            scenario['collision_distance'],
            scenario['relative_velocity'],
            avg_deviation,
            avg_fuel,
            avg_time,
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
    
    def reset(self):
        self.current_scenario = self.scenarios[np.random.randint(len(self.scenarios))]
        observation = self._get_observation()
        return observation, {'scenario_id': self.current_scenario['scenario_id']}
    
    def step(self, action):
        pareto = self.current_scenario['pareto_front']
        action = int(action) % len(pareto)
        selected_idx, all_preferences = self.fuzzy_decider.select_solution(pareto, epsilon=self.epsilon)
        suggested_solution = pareto[action]
        chosen_solution = pareto[selected_idx]
        if action == selected_idx:
            reward = 10.0
        else:
            dist = np.sqrt(
                ((suggested_solution['deviation'] - chosen_solution['deviation']) / 200.0) ** 2 +
                ((suggested_solution['fuel'] - chosen_solution['fuel']) / 50.0) ** 2 +
                ((suggested_solution['time'] - chosen_solution['time']) / 120.0) ** 2
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
            'all_preferences': all_preferences.tolist()
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
    env = MultiScenarioEnv(input_folder, fuzzy_decider, epsilon=0.15, use_genotype_encoding=encoding_type)
    model = PPO("MlpPolicy", env, verbose=1, batch_size=64, n_epochs=10, n_steps=2048, device='cpu',
                policy_kwargs=dict(net_arch=[dict(pi=[256,256], vf=[256,256])]))
    callback = MetricsCallback(verbose=1)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save("pareto_recommender_multi_scenario_model")
    print("Addestramento concluso. Modello salvato in 'pareto_recommender_multi_scenario_model'.")

if __name__ == "__main__":
    train_from_folder(input_folder="./data", total_timesteps=100000, encoding_type='stats')