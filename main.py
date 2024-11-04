import random
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

# Rozmery 2D priestoru
SPACE_SIZE = 5000

# Počet tried
NUM_CLASSES = 4

# Počet generovaných bodov pre testovanie
NUM_GENERATED_POINTS = 100 * NUM_CLASSES  # Nastavené na 10 000 bodov pre každú triedu

# Možné hodnoty pre parameter k
K = 3

# Počiatočné body
initial_points = {
    'R': [[-4500, -4400], [-4100, -3000], [-1800, -2400], [-2500, -3400], [-2000, -1400]],
    'G': [[+4500, -4400], [+4100, -3000], [+1800, -2400], [+2500, -3400], [+2000, -1400]],
    'B': [[-4500, +4400], [-4100, +3000], [-1800, +2400], [-2500, +3400], [-2000, +1400]],
    'P': [[+4500, +4400], [+4100, +3000], [+1800, +2400], [+2500, +3400], [+2000, +1400]]
}

# Funkcia na výpočet euklidovskej vzdialenosti medzi dvoma bodmi
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Funkcia na klasifikáciu nového bodu pomocou k-NN algoritmu s použitím k-d stromu
def classify(new_point, k, kd_tree, points):
    distances, indexes = kd_tree.query(new_point, k)
    neighbors = [points[i] for i in indexes]
    classes = [neighbor['class'] for neighbor in neighbors]
    return max(set(classes), key=classes.count)

# Funkcia na generovanie nových bodov
def generate_new_points(num_points, class_index):
    new_points = []
    for _ in range(num_points):
        if class_index == 0:  # R
            x = random.uniform(-SPACE_SIZE, +500) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
            y = random.uniform(-SPACE_SIZE, +500) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
        elif class_index == 1:  # G
            x = random.uniform(-500, SPACE_SIZE) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
            y = random.uniform(-SPACE_SIZE, +500) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
        elif class_index == 2:  # B
            x = random.uniform(-SPACE_SIZE, +500) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
            y = random.uniform(-500, SPACE_SIZE) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
        else:  # P
            x = random.uniform(-500, SPACE_SIZE) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
            y = random.uniform(-500, SPACE_SIZE) if random.random() < 0.99 else random.uniform(-SPACE_SIZE, SPACE_SIZE)
        new_points.append([x, y])
    return new_points

# Funkcia na zobrazenie grafu
def plot_classification(training_points, generated_points, results, k):
    colors = {'R': 'red', 'G': 'green', 'B': 'blue', 'P': 'purple'}
    
    # Plot the initial training points
    for point in training_points:
        plt.scatter(point['coords'][0], point['coords'][1], c=colors[point['class']], marker='o', edgecolor='black', s=100, label=f'Training {point["class"]}')
    
    # Plot the classified points
    for i, point in enumerate(generated_points):
        plt.scatter(point[0], point[1], c=colors[results[i]], marker='x', s=10, alpha=0.5)

    plt.title(f'Classification results for k = {k}')
    plt.xlim(-SPACE_SIZE, SPACE_SIZE)
    plt.ylim(-SPACE_SIZE, SPACE_SIZE)
    plt.grid(True)
    plt.show()

# Hlavná funkcia
def main():
    # Vyberte hodnotu k pre zobrazenie
    selected_k = 3  # Vyberte k z K_VALUES: 1, 3, 7, 15

    # Príprava trénovacích bodov
    training_points = []
    coords_list = []
    for class_name, coords_list_class in initial_points.items():
        for coords in coords_list_class:
            training_points.append({'coords': coords, 'class': class_name})
            coords_list.append(coords)

    # Vytvorenie k-d stromu z tréningových bodov
    kd_tree = KDTree(coords_list)
    
    # Generovanie bodov na klasifikáciu
    generated_points = []
    for class_index in range(NUM_CLASSES):
        new_points = generate_new_points(NUM_GENERATED_POINTS // NUM_CLASSES, class_index)
        generated_points.extend(new_points)

    # Klasifikácia pomocou vybranej hodnoty k
    results = []
    for point in generated_points:
        classification = classify(point, selected_k, kd_tree, training_points)
        results.append(classification)
    
    # Zobrazenie grafu pre vybranú hodnotu k
    plot_classification(training_points, generated_points, results, selected_k)

    # Zobrazenie výsledkov - počet klasifikovaných bodov pre každú triedu
    count_R = results.count('R')
    count_G = results.count('G')
    count_B = results.count('B')
    count_P = results.count('P')
    print(f"Results for k = {selected_k}:")
    print(f"R: {count_R}, G: {count_G}, B: {count_B}, P: {count_P}")
    print()

# Spustenie hlavnej funkcie
if __name__ == "__main__":
    main()
