import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Define initial points with coordinates and class labels
initial_points = [
    {"coords": [-4500, -4400], "class": "R"},
    {"coords": [-4100, -3000], "class": "R"},
    {"coords": [-1800, -2400], "class": "R"},
    {"coords": [-2500, -3400], "class": "R"},
    {"coords": [-2000, -1400], "class": "R"},
    {"coords": [4500, -4400], "class": "G"},
    {"coords": [4100, -3000], "class": "G"},
    {"coords": [1800, -2400], "class": "G"},
    {"coords": [2500, -3400], "class": "G"},
    {"coords": [2000, -1400], "class": "G"},
    {"coords": [-4500, 4400], "class": "B"},
    {"coords": [-4100, 3000], "class": "B"},
    {"coords": [-1800, 2400], "class": "B"},
    {"coords": [-2500, 3400], "class": "B"},
    {"coords": [-2000, 1400], "class": "B"},
    {"coords": [4500, 4400], "class": "P"},
    {"coords": [4100, 3000], "class": "P"},
    {"coords": [1800, 2400], "class": "P"},
    {"coords": [2500, 3400], "class": "P"},
    {"coords": [2000, 1400], "class": "P"},
]

# Define k-d tree node
class KDTreeNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

# Build k-d tree from points
def build_kdtree(points, depth=0):
    if not points:
        return None
    k = len(points[0]["coords"])  # Dimension of points
    axis = depth % k
    points.sort(key=lambda p: p["coords"][axis])
    median = len(points) // 2
    return KDTreeNode(
        point=points[median],
        axis=axis,
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1 :], depth + 1),
    )

# k-NN search in k-d tree
def knn_search(node, target, k, depth=0, best_neighbors=None):
    if node is None:
        return best_neighbors
    if best_neighbors is None:
        best_neighbors = []

    axis = node.axis
    dist = np.linalg.norm(np.array(node.point["coords"]) - np.array(target))
    if len(best_neighbors) < k or dist < best_neighbors[0][0]:
        best_neighbors.append((dist, node.point))
        best_neighbors.sort(reverse=True, key=lambda x: x[0])
        if len(best_neighbors) > k:
            best_neighbors.pop(0)

    next_branch = None
    opposite_branch = None
    if target[axis] < node.point["coords"][axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    knn_search(next_branch, target, k, depth + 1, best_neighbors)
    if len(best_neighbors) < k or abs(target[axis] - node.point["coords"][axis]) < best_neighbors[0][0]:
        knn_search(opposite_branch, target, k, depth + 1, best_neighbors)

    return best_neighbors

# Classification function
def classify(x, y, k, kdtree_root, points):
    neighbors = knn_search(kdtree_root, [x, y], k)
    nearest_classes = [neighbor[1]["class"] for neighbor in neighbors]
    most_common_class = Counter(nearest_classes).most_common(1)[0][0]
    points.append({"coords": [x, y], "class": most_common_class})
    kdtree_root = build_kdtree(points)  # Rebuild tree after each addition
    return most_common_class, kdtree_root

# Generate a random point for the specified class label with probabilistic coordinates
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

# Parameters and point generation
num_points_per_class = 1000
k = 7
kdtree_root = build_kdtree(initial_points)
generated_points = []
class_labels = ["R", "G", "B", "P"]

for _ in range(num_points_per_class):
    for label in class_labels:
        x, y = generate_point(label)
        point_class, kdtree_root = classify(x, y, k, kdtree_root, initial_points)
        generated_points.append({"coords": [x, y], "class": point_class})

# Visualization
all_points = initial_points + generated_points
x_coords = [point["coords"][0] for point in all_points]
y_coords = [point["coords"][1] for point in all_points]
colors = [point["class"] for point in all_points]
color_map = {"R": "red", "G": "green", "B": "blue", "P": "purple"}
point_colors = [color_map[cls] for cls in colors]

plt.figure(figsize=(10, 10))
plt.scatter(x_coords, y_coords, c=point_colors, s=5, alpha=0.7)
plt.xlim(-5000, 5000)
plt.ylim(-5000, 5000)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f'Classification results for k = {k}, Generated points: {num_points_per_class * len(class_labels)}')
plt.text(-4800, 4500, f"Value of k: {k}", fontsize=12, color='black')
plt.show()
