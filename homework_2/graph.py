import numpy as np 
from numpy.linalg import eig,inv
import networkx as nx 
import matplotlib.pyplot as plt
import sympy as sp

'''
These two function are just helper for plotting
'''
def rounding(list,digit):
    rounded_list = []
    for i in range(len(list)):
        rounded_list.append(round(list[i],digit))    
    return np.array(rounded_list)

def latex(matrix):
    return sp.Matrix(matrix)

'''
Class to help computing some graph characteristics. 
Some of them are computed coding the entire algorithm other are simply called from networkx package.
'''
class Graph():

    def __init__(self,W):
        self.weight_matrix = W
        self.size = W.shape[0]

    def in_degree(self):
        return self.weight_matrix @ np.ones(self.size)
    
    def out_degree(self):
        return self.weight_matrix.T @ np.ones(self.size)
    
    def average_degree(self):
        return 1/self.size * np.ones(self.size).T @ self.weight_matrix @ np.ones(self.size)
    
    def degree_matrix(self):
        return np.diag(self.in_degree())
    
    def laplacian_matrix(self):
        return self.degree_matrix() - self.weight_matrix
    
    def normalized_matrix(self):
        D  = self.degree_matrix()
        return inv(D)@ self.weight_matrix
    
    def invariant_centrality(self):
        eigval,eigvec = eig(self.normalized_matrix().T)
        for i in range(len(eigval)):
            if round(eigval[i],3) == 1:
                return eigvec[:,i]/np.sum(eigvec[:,i])
            
    def bonacich_centrality(self,mu,beta,step):
        alpha = 1 - beta
        pi = mu
        for i in range(step):
            pi = alpha * self.normalized_matrix().T @ pi + beta * mu
        return pi
    
        
    def neighbor(self,node):
        neighbor = []
        for i in range(self.size):
            if self.weight_matrix[node,i] != 0:
                neighbor.append(i)
        return neighbor

    #function to compute Dijkstra algorithm returning the cost for each node and the previous node with less cost
    def dijkstra(self,start):
        prev = {}
        cost = {}
        queue = []

        for vertex in range(self.size):
            cost[vertex] = np.inf
            prev[vertex] = None
            queue.append(vertex)
        cost[start] = 0

        while queue:
            q = []
            distances = []
            for vertex in queue:
                q.append(vertex)
                distances.append(cost[vertex])
            i = np.argmin(distances)
            u = q[i] 
            queue.remove(u)

            for v in self.neighbor(u):
                alt = cost[u] + self.weight_matrix[u,v]
                if alt < cost[v]:
                    cost[v] = alt
                    prev[v] = u

        return cost,prev

    # using returns from Dijkstra function, this function
    # compute the shortest path and respective cost from a starting node to a end one
    def dist(self,start,target):
        cost,prev = self.dijkstra(start)
        shortest_path = []
        u = target          
        while prev[u] != None:
            shortest_path.append(u)
            u = prev[u]
        shortest_path.reverse()
        return cost[target], shortest_path
    
    def wisdom_of_crowd(self, theta, noise):
        initial_state = np.ones(len(noise())) * theta + noise 
        consensus = self.invariant_centrality().T @ initial_state
        return consensus*np.ones(len(noise))/self.size


    def closeness_centrality(self,node):
        distances, _= self.dijkstra(node)
        distances_sum = 0
        for i in range(len(distances)):
            distances_sum += distances[i]
        return self.size/distances_sum
    
    def betweenness_centrality(self):
        graph = nx.from_numpy_array(self.weight_matrix,create_using=nx.MultiDiGraph)
        graph = graph.to_undirected()
        bc = nx.betweenness_centrality(graph)
        betweenness_centrality = []
        for i in range(self.size):
            betweenness_centrality.append(bc[i])
        return betweenness_centrality
    
    def draw_graph(self):
        graph = nx.from_numpy_array(self.weight_matrix,create_using=nx.MultiDiGraph)
        nx.draw(graph)
        plt.draw()

    def number_connected_components(self):
        graph = nx.from_numpy_array(self.weight_matrix,create_using=nx.MultiDiGraph)
        graph = graph.to_undirected()
        return nx.number_connected_components(graph)
        
    def number_strongly_connected_components(self):
      graph = nx.from_numpy_array(self.weight_matrix,create_using=nx.MultiDiGraph)
      return nx.number_strongly_connected_components(graph)
    
        