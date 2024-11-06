import math
import random
import time
from collections import namedtuple, Counter
import plotly.graph_objs as go

# Definícia bodu pre KD-Tree
Point = namedtuple("Point", ["x", "y", "label"])

k = 15  # Počet susedov, z ktorých vyberáme
num_points = 10000  # Počet bodov pre triedu

# Implementácia vlastného min-heapu
class MinHeap:
    def __init__(self):
        self.data = []

    def push(self, item):
        self.data.append(item)
        self._sift_up(len(self.data) - 1)

    def pop(self):
        if len(self.data) == 1:
            return self.data.pop()
        root = self.data[0]
        self.data[0] = self.data.pop()
        self._sift_down(0)
        return root

    def replace(self, item):
        root = self.data[0]
        self.data[0] = item
        self._sift_down(0)
        return root

    def pushpop(self, item):
        if self.data and item > self.data[0]:
            item, self.data[0] = self.data[0], item
            self._sift_down(0)
        return item

    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        while idx > 0 and self.data[idx] < self.data[parent]:
            self.data[idx], self.data[parent] = self.data[parent], self.data[idx]
            idx = parent
            parent = (idx - 1) // 2

    def _sift_down(self, idx):
        child = 2 * idx + 1
        while child < len(self.data):
            right = child + 1
            if right < len(self.data) and self.data[right] < self.data[child]:
                child = right
            if self.data[idx] <= self.data[child]:
                break
            self.data[idx], self.data[child] = self.data[child], self.data[idx]
            idx = child
            child = 2 * idx + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# KD-Tree štruktúra
class KDNode:
    def __init__(self, point, axis):
        self.point = point
        self.axis = axis
        self.left = None
        self.right = None

class KDTree:
    def __init__(self, points):
        self.root = self.build_tree(points, depth=0)

    def build_tree(self, points, depth):
        if not points:
            return None

        axis = depth % 2
        points.sort(key=lambda point: (point.x, point.y)[axis])
        median = len(points) // 2

        node = KDNode(points[median], axis)
        node.left = self.build_tree(points[:median], depth + 1)
        node.right = self.build_tree(points[median + 1:], depth + 1)

        return node

    def add_point(self, point):
        def _insert(root, point, depth):
            if root is None:
                return KDNode(point, depth % 2)

            axis = root.axis
            if (point.x, point.y)[axis] < (root.point.x, root.point.y)[axis]:
                root.left = _insert(root.left, point, depth + 1)
            else:
                root.right = _insert(root.right, point, depth + 1)
            return root

        self.root = _insert(self.root, point, depth=0)

    def nearest_neighbors(self, target, k):
        best = MinHeap()

        def _search(root):
            if root is None:
                return

            dist = euclidean_distance(target, root.point)
            if len(best) < k:
                best.push((-dist, root.point))
            elif dist < -best[0][0]:
                best.replace((-dist, root.point))

            axis = root.axis
            diff = (target.x, target.y)[axis] - (root.point.x, root.point.y)[axis]
            close, away = (root.left, root.right) if diff < 0 else (root.right, root.left)

            _search(close)
            if len(best) < k or abs(diff) < -best[0][0]:
                _search(away)

        _search(self.root)
        return [point.label for _, point in sorted(best.data, reverse=True)]

def euclidean_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
 

# Počiatočné body s triedami ako dictionary
initial_points = {
    'R': [Point(-4500, -4400, 'R'), Point(-4100, -3000, 'R'), Point(-1800, -2400, 'R'), 
          Point(-2500, -3400, 'R'), Point(-2000, -1400, 'R')],
    'G': [Point(4500, -4400, 'G'), Point(4100, -3000, 'G'), Point(1800, -2400, 'G'), 
          Point(2500, -3400, 'G'), Point(2000, -1400, 'G')],
    'B': [Point(-4500, 4400, 'B'), Point(-4100, 3000, 'B'), Point(-1800, 2400, 'B'), 
          Point(-2500, 3400, 'B'), Point(-2000, 1400, 'B')],
    'P': [Point(4500, 4400, 'P'), Point(4100, 3000, 'P'), Point(1800, 2400, 'P'), 
          Point(2500, 3400, 'P'), Point(2000, 1400, 'P')]
}

all_points = [point for points in initial_points.values() for point in points]
kd_tree = KDTree(all_points)

def classify(X, Y):
    target = Point(X, Y, None)
    k_nearest_labels = kd_tree.nearest_neighbors(target, k)
    assigned_label = Counter(k_nearest_labels).most_common(1)[0][0]
    kd_tree.add_point(Point(X, Y, assigned_label))
    return assigned_label

def generate_point(label):
    if label == 'R':
        if random.random() < 0.99:
            X = random.randint(-5000, 500)
            Y = random.randint(-5000, 500)
        else:
            X = random.randint(-5000, 5000)
            Y = random.randint(-5000, 5000)
    
    elif label == 'G':
        if random.random() < 0.99:
            X = random.randint(-500, 5000)
            Y = random.randint(-5000, 500)
        else:
            X = random.randint(-5000, 5000)
            Y = random.randint(-5000, 5000)
    
    elif label == 'B':
        if random.random() < 0.99:
            X = random.randint(-5000, 500)
            Y = random.randint(-500, 5000)
        else:
            X = random.randint(-5000, 5000)
            Y = random.randint(-5000, 5000)
    
    elif label == 'P':
        if random.random() < 0.99:
            X = random.randint(-500, 5000)
            Y = random.randint(-500, 5000)
        else:
            X = random.randint(-5000, 5000)
            Y = random.randint(-5000, 5000)
    
    return X, Y

def run():
    start_time = time.time()
    correct = 0
    correct_per_class = {'R': 0, 'G': 0, 'B': 0, 'P': 0}
    labels_cycle = ['R', 'G', 'B', 'P'] * num_points
    
    for label in labels_cycle:
        X, Y = generate_point(label)
        predicted_label = classify(X, Y)
        
        if predicted_label == label:
            correct += 1
            correct_per_class[label] += 1
    
    accuracy = correct / (num_points * 4) * 100
    print(f"Overall accuracy for k={k}: {accuracy:.2f}%")

    for label in correct_per_class:
        class_accuracy = correct_per_class[label] / num_points * 100
        print(f"Accuracy for class {label}: {class_accuracy:.2f}% ({correct_per_class[label]} out of {num_points})")
    
    runtime = time.time() - start_time
    print(f"Runtime: {runtime:.2f} seconds")
    
    return accuracy, correct_per_class, runtime

def visualize_classification_plotly():
    points = []
    labels_cycle = ['R', 'G', 'B', 'P'] * num_points

    for label in labels_cycle:
        X, Y = generate_point(label)
        predicted_label = classify(X, Y)
        points.append((X, Y, predicted_label))

    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    labels = [p[2] for p in points]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x_coords, y=y_coords,
        mode='markers',
        marker=dict(
            color=[{'R': 'red', 'G': 'green', 'B': 'blue', 'P': 'purple'}[label] for label in labels],
            size=8,
            
        )
    ))
    fig.update_layout(width=800, height=800, title="k-NN Classification")
    fig.show()

def main():
    run()
    visualize_classification_plotly()

# Spustenie hlavnej funkcie
if __name__ == "__main__":
    main()