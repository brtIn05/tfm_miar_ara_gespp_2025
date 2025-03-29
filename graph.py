import logging
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime

class Graph:
    def __init__(self):
        self.nodes = set()
        self.source = None # Initialize it later with padding
        self.sink = None
        self.node_coords = {}       # node: (x_i, y_i, deltai)
        self.arcs = {}             # (i, j): cost
        self.neighbours = {}         # i: set(j)
        self.clusters = {}          # cluster_id: set(node)
        self.node_clusters = {}     # node: set(cluster_id)
        self.cluster_profits = {}   # cluster_id: profit

    def add_node(self, node, node_info):
        '''Add nodes to the graph defined by coordinates x, y and delta.'''
        self.nodes.add(node)
        self.node_coords[node] = node_info
        self.neighbours.setdefault(node, set())

    def add_cluster(self, raw_clusters,  padding): #cluster_id, nodes, profit,
        '''Add clusters using list containing tuples of profit plus binary string'''

        # self.clusters[cluster_id] = set(nodes)
        # self.cluster_profits[cluster_id] = profit
        # for node in nodes:
        #     self.node_clusters.setdefault(node, set()).add(cluster_id) # Si es la primera vez anadimos el nodo y la entrada

        for id_cluster, cluster_info in enumerate(raw_clusters):
            # Calculate id of the cluster
            id_cluster = "C"+str(id_cluster+1).rjust(padding, '0')

            # Cluster info contains profit and binary string
            cluster_profit, cluster_binary_string = cluster_info

            # Add profits to dictionary
            self.cluster_profits[id_cluster] =  int(cluster_profit)

            # Add each node in a set associated with each cluster ID
            self.clusters.setdefault(id_cluster,set())

            for index, binary_element in enumerate(cluster_binary_string):

                if binary_element == "1":
                    node = str(index + 2).rjust(padding, '0')
                    self.clusters[id_cluster].add(node)

            for key, values in self.clusters.items():
                for value in values:
                    self.node_clusters.setdefault(value,set()).add(key)  
    

    def compute_arcs_cost(self):
        '''Add cost for all graph arcs'''
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    cost = self.calculate_arc_cost(i, j)
                    self.arcs[(i, j)] = cost
                    self.neighbours[i].add(j)
        arcs_costs_dict = {str(k):v for k, v in self.arcs.items()}
        return arcs_costs_dict


    def calculate_arc_cost(self, i, j):
        '''Computes arc cost using tuple of two arguments consisting on tuple of node coordinates and accessing delta of each node. \n
        Formula used is ceil: \n (100*(math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2))- (delta_i / 2) - (delta_j / 2)).
        '''
        x_i, y_i, delta_i = self.node_coords[i]
        x_j, y_j, delta_j = self.node_coords[j]
        arc_cost =  math.ceil(100*(math.sqrt((x_i - x_j)**2 + (y_i - y_j)**2))- (delta_i / 2) - (delta_j / 2))
        return arc_cost

    def load_graph_data(self, filename):
        '''Initialize complete graph''' 
        logging.info(f"########## READING GRAPH ##########")  
        start_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        filename_clean = filename[filename.find('/')+1:len(filename)]

        with open(filename, 'r') as f:
            lines = f.readlines()

            nodes_number = int(lines[0].split('=')[1].strip())
            self.sink = str(nodes_number+2)
            clusters_number = int(lines[1].split('=')[1].strip())
            padding = len(str(nodes_number))
            self.source = "1".rjust(padding, '0')

            # Read node data starting from the third line
            for id_node,line in enumerate(lines[2:nodes_number+4]):
                parts = line.strip().split()
                x_i, y_i, delta_i = map(int, parts)
                # Add nodes read in the source file
                self.add_node( str(id_node+1).rjust(padding, '0'),(x_i, y_i, delta_i))

            # Once nodes with coordinates are loaded arcs cost can be calculated
            arcs_costs = self.compute_arcs_cost()

            # Read clusters data and store it in list of tuples (profit, binary string)
            raw_clusters = []
            cluster_lines = lines[nodes_number+4:]
            for line in cluster_lines:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 2:
                        profit = float(parts[0])
                        binary_string = parts[1]
                        raw_clusters.append((profit, binary_string))


            # Call function to create clusters data types
            self.add_cluster(raw_clusters, padding)


        logging.info(f"\n========== Run file {filename_clean} Started at: {start_time} ==========\n")
        logging.info(f"## Graph {filename_clean} loaded correctly. \n\n  ++ Size = {nodes_number}\n  ++ Clusters = {clusters_number}\n")
        logging.info(f"  -> Nodes: {self.nodes}\n")
        logging.info(f"  -> Nodes coordinates: {self.node_coords}\n")
        logging.info(f"  -> Nodes neighbours: {self.neighbours}\n")
        logging.info(f"  -> Nodes associated clusters: {self.node_clusters}\n")
        logging.info(f"  -> Clusters: {self.clusters}\n")
        logging.info(f"  -> Clusters Profits: {self.cluster_profits}\n")
        logging.info("  -> Arcs costs:")
        logging.info(json.dumps(arcs_costs, indent = 4))

        return nodes_number, clusters_number, padding
    
    def plot_nodes(self, filename, padding):
        '''Plots all nodes in graph. Source and sink nodes are highlighted in red'''
        fig, ax = plt.subplots()
        for node in self.nodes:
            x, y, delta = self.node_coords[node]
            if node == '1'.rjust(padding, '0') or node == str(len(self.nodes)).rjust(padding, '0'):
                ax.plot(x, y, 'ro')  
                # Annotate the circle with its node id
                ax.text(x+2, y+2, node, fontsize=7, ha='right')
            else:
                ax.plot(x, y, 'bo')  
                # Annotate the circle with its node id
                ax.text(x+2, y+2, node, fontsize=7, ha='right')

            # circle = plt.Circle((x, y), delta/2, color='C0', fill=True, alpha=0.5)
            # ax.add_patch(circle)

        # Display the plot test
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Graph representation for instance {filename[filename.find('/')+1:len(filename)]}')
        plt.grid(True)

    def plot_path(self, path, filename, padding):
        """
        Plots the graph nodes (with source and sink highlighted) and overlays a path.
        """
        
        fig, ax = plt.subplots()
        
        for node in self.nodes:
            x, y, delta = self.node_coords[node]
            if node == '1'.rjust(padding, '0') or node == str(len(self.nodes)).rjust(padding, '0'):
                ax.plot(x, y, 'ro')  
                # Annotate the circle with its node id
                ax.text(x+2, y+2, node, fontsize=7, ha='right')
            else:
                ax.plot(x, y, 'bo')  
                # Annotate the circle with its node id
                ax.text(x+2, y+2, node, fontsize=7, ha='right')
        

        padded_path = [n.rjust(padding, '0') for n in path]
        
        for i in range(len(padded_path) - 1):
            node1 = padded_path[i]
            node2 = padded_path[i+1]
            x1, y1, _ = self.node_coords[node1]
            x2, y2, _ = self.node_coords[node2]
            ax.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Graph Path for instance {filename[filename.find('/')+1:len(filename)]}')
        plt.grid(True)

        return 