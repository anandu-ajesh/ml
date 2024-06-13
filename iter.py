from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def iterative_deepening_search(self, start, goal):
        for depth in range(self.V):
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if node == goal:
                    return path
                for neighbor in self.graph[node]:
                    if neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))
        return None

g = Graph(8)
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(0, 3)
g.addEdge(1, 1)
g.addEdge(2, 2)
g.addEdge(3, 4)
g.addEdge(3, 5)
g.addEdge(4, 6)
g.addEdge(5, 7)

path = g.iterative_deepening_search(0, 6)
if path:
    print(f"Path found: {path}")
    print(f"Depth of search: {len(path) - 1}")
else:
    print("No path found.")