from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Definícia počiatočných bodov podľa zadania
initial_points = [
    # Red (R) points
    {"coords": [-4500, -4400], "class": "R"},
    {"coords": [-4100, -3000], "class": "R"},
    {"coords": [-1800, -2400], "class": "R"},
    {"coords": [-2500, -3400], "class": "R"},
    {"coords": [-2000, -1400], "class": "R"},
    
    # Green (G) points
    {"coords": [4500, -4400], "class": "G"},
    {"coords": [4100, -3000], "class": "G"},
    {"coords": [1800, -2400], "class": "G"},
    {"coords": [2500, -3400], "class": "G"},
    {"coords": [2000, -1400], "class": "G"},
    
    # Blue (B) points
    {"coords": [-4500, 4400], "class": "B"},
    {"coords": [-4100, 3000], "class": "B"},
    {"coords": [-1800, 2400], "class": "B"},
    {"coords": [-2500, 3400], "class": "B"},
    {"coords": [-2000, 1400], "class": "B"},
    
    # Purple (P) points
    {"coords": [4500, 4400], "class": "P"},
    {"coords": [4100, 3000], "class": "P"},
    {"coords": [1800, 2400], "class": "P"},
    {"coords": [2500, 3400], "class": "P"},
    {"coords": [2000, 1400], "class": "P"},
]

# Extrahovanie súradníc a vytvorenie k-d stromu
coordinates = [point["coords"] for point in initial_points]
kdtree = cKDTree(coordinates)

# Funkcia na klasifikáciu bodu pomocou k-NN
def classify(x, y, k, kdtree, points):
    distances, indices = kdtree.query([x, y], k=k)
    nearest_classes = [points[i]["class"] for i in indices]
    most_common_class = Counter(nearest_classes).most_common(1)[0][0]
    points.append({"coords": [x, y], "class": most_common_class})
    kdtree = cKDTree([point["coords"] for point in points])
    return most_common_class, kdtree

# Funkcia na generovanie bodu pre danú triedu s pravdepodobnostnými podmienkami
def generate_point(class_label):
    if np.random.rand() < 0.99:
        if class_label == "R":
            x = np.random.uniform(-5000, 500)
            y = np.random.uniform(-5000, 500)
        elif class_label == "G":
            x = np.random.uniform(-500, 5000)
            y = np.random.uniform(-5000, 500)
        elif class_label == "B":
            x = np.random.uniform(-5000, 500)
            y = np.random.uniform(-500, 5000)
        elif class_label == "P":
            x = np.random.uniform(-500, 5000)
            y = np.random.uniform(-500, 5000)
    else:
        x = np.random.uniform(-5000, 5000)
        y = np.random.uniform(-5000, 5000)
    return [x, y]

# Parametre
num_points_per_class = 1000
k = 7
generated_points = []
class_labels = ["R", "G", "B", "P"]
for i in range(num_points_per_class):
    for label in class_labels:
        x, y = generate_point(label)
        point_class, kdtree = classify(x, y, k, kdtree, initial_points)
        generated_points.append({"coords": [x, y], "class": point_class})

# Vizualizácia
all_points = initial_points + generated_points
x_coords = [point["coords"][0] for point in all_points]
y_coords = [point["coords"][1] for point in all_points]
colors = [point["class"] for point in all_points]
color_map = {"R": "red", "G": "green", "B": "blue", "P": "purple"}
point_colors = [color_map[cls] for cls in colors]

plt.scatter(x_coords, y_coords, c=point_colors, s=5, alpha=0.7)
plt.xlim(-5000, 5000)
plt.ylim(-5000, 5000)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f'Classification results for k = {k}, Počet vygenerovaných bodov: {num_points_per_class * len(class_labels)}')
plt.text(-4800, 4500, f"Hodnota k: {k}", fontsize=12, color='black')
plt.show()