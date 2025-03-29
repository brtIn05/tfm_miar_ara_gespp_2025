import logging
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime
import random

class ACO_gespp:
    def __init__(self, graph, num_ants, num_iterations,
                 alpha, beta,evaporation_rate, Q,
                 maxmin_init_cost, tau_min_scaler, use_maxmin,
                 heuristic_type="Cost", profit_gamma=1.0, profit_epsilon=1.0, Qstrategy=True):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha # Pheromone influence
        self.beta = beta # Heuristic influence
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.Qstrategy = Qstrategy
        self.min_arc_cost = min(self.graph.arcs.values()) # Min arc cost (“Normalize” the heuristic for negative arc costs)

        # Heuristic selection
        self.heuristic_type = heuristic_type    # Select Heuristic to apply
        self.profit_gamma = profit_gamma        # Gamma value to balance Profit vs ArcCost
        self.profit_epsilon = profit_epsilon    # Small constant for profit-based heuristic

        # Modification of MAXMIN algorithm
        self.use_maxmin = use_maxmin  # Choose MAXMIN mode
        if self.use_maxmin:
            # MAX–MIN mode: set pheromone limits.
            self.tau_max = self.Q / maxmin_init_cost
            self.tau_min = self.tau_max / tau_min_scaler
            self.pheromone = {arc: self.tau_max for arc in self.graph.arcs}
        else:
            # Standard ACO: initialize pheromone on all arcs to 1.0.
            self.pheromone = {arc: 1.0 for arc in self.graph.arcs}



    def run(self):
        '''ACO algorithm run. Loop for specific number of iterations for a given number of ants.'''
        logging.info(f"########## RUNNING ACO MODEL ##########")  
        # Intitialize solution and cost
        best_path = None
        best_cost = float('inf')
        progress = {}

        # Main loop for the run. Iteration for a given number of solutions.
        for iteration in range(self.num_iterations):
            # Restart solutions in new iteration
            solutions = []
            for ant in range(self.num_ants):
                path, cost, total_arc_cost, total_profit, visited_clusters = self.ant_build_solution()
                # Append solution to list of solutions
                if path is not None:
                    solutions.append((path, cost))
                    # Update solution if better 
                    if cost < best_cost:
                        best_path = path
                        best_cost = cost
                        best_total_arc_cost = total_arc_cost
                        best_total_profit = total_profit
                        best_visited_clusters = visited_clusters

            # Update pheromones once all ants are finished
            self.update_pheromones(solutions, best_cost)
            progress[iteration + 1] = cost      
            logging.info(f"Iteration {iteration + 1}: Cost of latest solution: {cost} Best cost so far: {best_cost}")

        best_clusters_coverage = round((len(best_visited_clusters)/len(list(self.graph.clusters.keys())))*100,2)
        
        logging.info(f"ACO Path: {best_path}")
        logging.info(f"ACO Objective function: {best_cost}")

        logging.info("#### ACO Solution details ####")
        logging.info("    Selected arcs in the solution:")

        logging.info(f"    ACO Path cost (only arcs): {best_total_arc_cost}")  
        logging.info(f"    ACO Clusters profit recovered: {best_total_profit}")
        logging.info(f"    ACO Clusters visited: {best_visited_clusters}")
        logging.info(f"    ACO Clusters usage: {best_clusters_coverage}%")

        return best_path, best_cost, best_total_arc_cost, best_total_profit, best_clusters_coverage, progress

    def ant_build_solution(self):
        '''
        Individual Ant building a solution.
        Each Ant has a probability of selecting a node based on (pheromone^alpha)*(heuristic^beta).
        '''
        current = self.graph.source
        path = [current]
        nodes_visited = set(path)

        # Obteain clusters already visited in the path
        # In each iteration, for each new node visited, add clusters in set
        if self.heuristic_type == "ProfitCost":
            path_visited_clusters = set()
            if current in self.graph.node_clusters: # Verify node has cluster associated
                path_visited_clusters.update(self.graph.node_clusters[current])

        # Loop to build each ant path
        while current != self.graph.sink:
            # Candidate nodes can be choose based on not yet selected in path
            node_candidates_for_ant = [node for node in self.graph.neighbours[current] if node not in nodes_visited]
            # If not connected because not fully connected graph
            if not node_candidates_for_ant:
                return None, None

            # Calculate probabilities to next candidate node
            prob_per_node = []
            prob_sum_all_nodes = 0.0
            for j in node_candidates_for_ant:
                tau = self.pheromone[(current, j)]
                cost = self.graph.arcs[(current, j)]
                # Normalize negative arc cost:
                normalize_cost = cost - self.min_arc_cost + 1

                if self.heuristic_type == "ProfitCost" and len(path_visited_clusters) == len(self.graph.clusters):
                    # Calculate potential profit for candidate node j
                    potential_profit = 0
                    if j in self.graph.node_clusters:
                        # Loop thourhg all cluster associated to candidate node and sum potential Profit if not already covered
                        for cluster in self.graph.node_clusters[j]:
                            if cluster not in path_visited_clusters:
                                potential_profit += self.graph.cluster_profits[cluster]
                    # ProfitCost heuristic:
                    heuristic = ((potential_profit + self.profit_epsilon) / (normalize_cost + self.graph.arcs[(current, self.graph.sink)])) ** self.profit_gamma
                else:
                    # Cost heuristic
                    heuristic = 1.0 / normalize_cost

                # 
                prob_node_value = (tau ** self.alpha) * (heuristic ** self.beta)
                prob_per_node.append(prob_node_value)
                prob_sum_all_nodes += prob_node_value

            
            # In case all prob_per_node are 0, choose uniformly
            if prob_sum_all_nodes == 0:
                prob_per_node = [1.0 for _ in node_candidates_for_ant]
                prob_sum_all_nodes = len(node_candidates_for_ant)
            
            # Normalize each probability for going to next node
            prob_per_node = [prob / prob_sum_all_nodes for prob in prob_per_node]
            # Randomly select the next node in path
            next_node = random.choices(node_candidates_for_ant, weights=prob_per_node, k=1)[0]
            
            # Update path and move to next node
            path.append(next_node)
            nodes_visited.add(next_node)
            current = next_node

        # Compute objective function
        objective_function_cost, total_arc_cost, total_profit, visited_clusters = self.evaluate_path(path)
        return path, objective_function_cost, total_arc_cost, total_profit, visited_clusters

    def evaluate_path(self, path):
        '''
        Compute the objective function of the GESPP problem.
        Sum all the arc costs in the path.
        Identify clusters visited and sum profits.
        '''
        # Compute sum of path arc costs
        total_arc_cost = 0
        for i in range(len(path) - 1):
            total_arc_cost += self.graph.arcs[(path[i], path[i + 1])]

        # Evaluate which clusters have been visited in the path and compute
        # sum of profits
        visited_clusters = set()
        for node in path:
            if node in self.graph.node_clusters:
                visited_clusters.update(self.graph.node_clusters[node])
        
        # Compute objective function
        total_profit = sum(self.graph.cluster_profits[cluster] for cluster in visited_clusters)
        objective_function_cost = total_arc_cost - total_profit
        return objective_function_cost, total_arc_cost, total_profit, visited_clusters

    def update_pheromones(self, solutions, best_solution_cost):
        '''Apply evaporation and deposit pheromone'''

        # Evaporate pheromone on every arc
        for arc in self.pheromone:
            self.pheromone[arc] = self.pheromone[arc] * (1 - self.evaporation_rate)

        if self.use_maxmin:
            # MAX–MIN mode: deposit only using the best solution from the iteration.
            best_solution_in_iter, best_cost_in_iter = min(solutions, key=lambda s: s[1])
            deposit = self.Q  # Only best solution is used so (cost - best_solution_cost + 1) = 1
            for i in range(len(best_solution_in_iter) - 1):
                arc = (best_solution_in_iter[i], best_solution_in_iter[i + 1])
                self.pheromone[arc] += deposit
            
            # Enforce pheromone bounds.
            for arc in self.pheromone:
                if self.pheromone[arc] > self.tau_max:
                    self.pheromone[arc] = self.tau_max
                elif self.pheromone[arc] < self.tau_min:
                    self.pheromone[arc] = self.tau_min
        else:
            # Standard ACO Ant System mode: deposit pheromones for every solution.
            for path, cost in solutions:
                if cost is None:
                    continue
                if self.Qstrategy:
                    deposit = self.Q / (cost - best_solution_cost + 1)
                else:
                    deposit = self.Q / (1/cost)
                for i in range(len(path) - 1):
                    arc = (path[i], path[i + 1])
                    self.pheromone[arc] += deposit

    def local_search_3opt(self, path, cost):
        """
        Use local search as post process of the ACO path.
        Local Search based on 3-opt moves
        """

        # Intilize variables. Path & cost from ACO considered as best befor LS
        path_improved = True
        best_path = path
        best_cost = cost
        
        while path_improved:
            path_improved = False
            # Loop over all path breaking 3 arcs in the path. 
            # Each loop iterates starting from the previous one
            for i in range(1, len(best_path) - 3):
                for j in range(i + 1, len(best_path) - 2):
                    for k in range(j + 1, len(best_path) - 1):
                        candidate_path, candidate_cost = self.move_3opt(best_path, i, j, k)
                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_path = candidate_path
                            path_improved = True
                            # Restart search after improvement
                            break
                    if path_improved:
                        break
                if path_improved:
                    break
        
        # Call evaluate path just for clusters coverage
        _, _, _, best_visited_clusters = self.evaluate_path(best_path)
        best_clusters_coverage = round((len(best_visited_clusters) / len(list(self.graph.clusters.keys()))) * 100, 2)
        return best_path, best_cost, best_clusters_coverage

    def move_3opt(self, path, i, j, k):
        """
        For sliced arcs i, j, k in the path, generate candidate path using reverse and swap moves.
        Returns a pathe if improved objective cost.
        """
        # Slice path cutting by arcs selected
        A, B, C, D = path[:i], path[i:j], path[j:k], path[k:]
        
        # List candidate reconnections (a subset of possible 3‑opt moves)
        candidate_paths = []
        candidate_paths.append(A + B[::-1] + C + D)         # Reverse segment B
        candidate_paths.append(A + B + C[::-1] + D)           # Reverse segment C
        candidate_paths.append(A + B[::-1] + C[::-1] + D)      # Reverse both B and C
        candidate_paths.append(A + C + B + D)                 # Swap segments B and C
        candidate_paths.append(A + C + B[::-1] + D)           # Swap with B reversed
        candidate_paths.append(A + C[::-1] + B + D)           # Swap with C reversed
        
        best_cost = self.evaluate_path(path)[0]
        best_candidate_path = path
        best_candidate_cost = best_cost
        for candidate_path in candidate_paths:
            cand_cost, _, _, _ = self.evaluate_path(candidate_path)
            if cand_cost < best_candidate_cost:
                best_candidate_cost = cand_cost
                best_candidate_path = candidate_path
        return best_candidate_path, best_candidate_cost
