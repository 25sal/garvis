import math
import random
import numpy as np
from shapely.geometry import LineString, Point, Polygon
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------
# UTILITY GEOMETRICHE
# ---------------------------------------------------
def generate_random_point(size):
    return random.uniform(0, size), random.uniform(0, size)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_border_intersection(x, y, angle, size):
    dx, dy = math.cos(angle), math.sin(angle)
    t_x_min = -x / dx if dx != 0 else float('inf')
    t_x_max = (size - x) / dx if dx != 0 else float('inf')
    t_y_min = -y / dy if dy != 0 else float('inf')
    t_y_max = (size - y) / dy if dy != 0 else float('inf')
    t_enter = max(min(t_x_min, t_x_max), min(t_y_min, t_y_max))
    t_exit = min(max(t_x_min, t_x_max), max(t_y_min, t_y_max))
    x_in, y_in = x + t_enter * dx, y + t_enter * dy
    x_out, y_out = x + t_exit * dx, y + t_exit * dy
    return (x_in, y_in), (x_out, y_out)

def sample_trajectory(traj, speed, interval):
    sampled = [traj.coords[0]]
    total_length = traj.length
    num_steps = int((total_length / speed) * 60 / interval)
    for i in range(1, num_steps + 1):
        d = speed * interval / 60 * i
        if d >= total_length:
            break
        p = traj.interpolate(d)
        sampled.append((p.x, p.y))
    return sampled

def check_max_turn_angle(path, max_angle_deg=25):
    coords = list(path.coords)
    if len(coords) < 3:
        return True
    max_angle_rad = math.radians(max_angle_deg)
    for i in range(1, len(coords) - 1):
        p0, p1, p2 = coords[i-1], coords[i], coords[i+1]
        v1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        v2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return False
        cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.acos(cos_theta)
        if angle > max_angle_rad:
            logger.debug(f"Turn angle {math.degrees(angle)} exceeds max {max_angle_deg} at point {p1}")
            return False
    return True

def is_valid_path(point_list, scenario):
    path = LineString(point_list)
    area_poly = Polygon([(0, 0), (0, scenario["area_size"]), (scenario["area_size"], scenario["area_size"]), (scenario["area_size"], 0)])
    if not area_poly.covers(path) or not check_max_turn_angle(path):
        logger.debug(f"Path is out of bounds {area_poly.covers(path)}   or exceeds max turn angle {check_max_turn_angle(path)}.")
        return False
    sampled = sample_trajectory(path, scenario["speed_early"], scenario["sampling_interval"])
    primary_sampled = sample_trajectory(LineString([scenario["ing_early"], scenario["usc_early"]]), scenario["speed_early"], scenario["sampling_interval"])
    for p1, p2 in zip(sampled, primary_sampled):
        if distance(p1, p2) < scenario["separation_min"]:
            logger.debug(f"Collision detected at points {p1} and {p2} with distance {distance(p1, p2)}")
            return False
    return True

'''
def is_valid_path(point_list, area_poly, primary_traj, separation_min, t_max, speed):
    path = LineString(point_list)
    if not area_poly.covers(path) or not check_max_turn_angle(path):
        return False
    sampled = sample_trajectory(path, speed, sampling_interval)
    primary_sampled = sample_trajectory(primary_traj, speed, sampling_interval)
    for p1, p2 in zip(sampled, primary_sampled):
        if distance(p1, p2) < separation_min:
            return False
    return True
'''
def compute_curvature(path):
    coords = list(path.coords)
    if len(coords) < 3:
        return 0
    angles = []
    for i in range(1, len(coords) - 1):
        p0, p1, p2 = coords[i - 1], coords[i], coords[i + 1]
        v1 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
        v2 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue
        angle = np.arccos(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1))
        angles.append(angle)
    return np.mean(angles) 


def compute_initial_bearing(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.atan2(dy, dx)
    return angle

def collision_detection(path, pcenter, radius, time_slot):
    # Define circle as buffered Point
    circle = Point(pcenter).buffer(radius)

    # Filter points by time interval [t_start, t_end]
    # filtered_points = [pt for pt in path if time_slot[0] <= pt[2] <= time_slot[1]]
    '''
    I do not consider the time to increase the probability of collision
    '''
    filtered_points = [pt for pt in path]
    if len(filtered_points) >= 2:
        sub_route = LineString([(pt[0], pt[1]) for pt in filtered_points])
        is_inside = sub_route.intersects(circle)
    else:
        is_inside = False  # Not enough points in interval to define route
    if is_inside:
        logger.debug(f"Collision detected")
    return is_inside