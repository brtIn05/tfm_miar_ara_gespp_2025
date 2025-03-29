import pulp
import math
import logging

def MIP_gespp(graph):
    """
    Solves GESPP instance based on MIP model
    """

    logging.info(f"########## RUNNING IP MODEL ##########")  
    model = pulp.LpProblem("MIP_GESPP", pulp.LpMinimize)

    # Decision Variables
    x_vars = {}
    for (i, j), cost in graph.arcs.items():
        if i != j:
            x_vars[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
    y_vars = {}
    for c in graph.cluster_profits:
        y_vars[c] = pulp.LpVariable(f"y_{c}", cat=pulp.LpBinary)


    # Objective Function
    model += (
        pulp.lpSum([graph.arcs[(i, j)] * x_vars[(i, j)] for (i, j) in x_vars])
        - pulp.lpSum([graph.cluster_profits[c] * y_vars[c] for c in y_vars])
    ), "TotalCost"

    # Constraints
    source = graph.source
    sink = graph.sink

    # 1. Start at the source.
    model += (
        pulp.lpSum([x_vars[(source, j)] for j in graph.neighbours[source]]) == 1), "Source_outflow"

    # 2. Finish at the sink.
    model += (
        pulp.lpSum([x_vars[(i, sink)] for i in graph.neighbours[sink]])== 1), "Sink_inflow"

    # 3. Flow conservation.
    for node in graph.nodes:
        if node not in [source, sink]:
            # Flow conservation: the number of arcs entering equals the number leaving.
            model += (
                pulp.lpSum([x_vars[(i, node)] for i in graph.neighbours[node]]) - pulp.lpSum([x_vars[(node, j)] for j in graph.neighbours[node]]) == 0
                ), f"Flow_conservation_{node}"
            
    # 4. Cycles.
    for node in graph.nodes:
        if node not in [source, sink]:
            # Relaxed cycle elimination: at most one arc enters the node.
            model += (
                pulp.lpSum([x_vars[(i, node)] for i in graph.neighbours[node]]) <= 1
            ), f"Subtour_elimination_{node}"

    # 4. Sink & source.

    model += (
        pulp.lpSum([x_vars[(i, source)] for i in graph.neighbours[source]]) == 0
    ), f"No_enter_source_elimination_{node}"
    model += (
        pulp.lpSum([x_vars[(sink, j)] for j in graph.neighbours[sink]]) == 0
    ), f"No_exit_sink_elimination_{node}"

    # 5. Cluster visit 
    for c, node_set in graph.clusters.items():
        model += (pulp.lpSum([pulp.lpSum([x_vars[(i, j)] for j in graph.neighbours[i] if (i, j) in x_vars]) for i in node_set if i in graph.nodes])
                >= y_vars[c]), f"ClusterActivation_{c}"
        

    # Solve the Model
    solver = pulp.PULP_CBC_CMD(msg=True)  # Using CBC (default PuLP solver)
    result = model.solve(solver)
    status = pulp.LpStatus[model.status]
    objective = pulp.value(model.objective)
    logging.info(f"IP Status: {status}")
    logging.info(f"IP Objective function: {objective}")

    # Extract the Solution
    path_arcs = [(i, j) for (i, j) in x_vars if pulp.value(x_vars[(i, j)]) > 0.5]
    logging.info("\n#### IP Solution details ####")
    # logging.info("\nSelected arcs in the solution:")
    # for arc in path_arcs:
    #     logging.info(f"Arc {arc} with cost {graph.arcs[arc]}")


    visited_clusters = [c for c in y_vars if pulp.value(y_vars[c]) > 0]
    total_arc_cost = sum(graph.arcs[(i, j)] * pulp.value(x_vars[(i, j)]) for (i, j) in x_vars)
    total_profit = sum(graph.cluster_profits[cluster] for cluster in visited_clusters)
    clusters_coverage = (len(visited_clusters)/len(list(graph.clusters.keys())))*100

    logging.info(f"IP Path cost (only arcs): {total_arc_cost}")  
    logging.info(f"IP Clusters profit recovered: {total_profit}")
    logging.info(f"IP Clusters visited: {visited_clusters}")
    logging.info(f"IP Clusters usage: {clusters_coverage}%")


    return status, objective, path_arcs, total_arc_cost, total_profit, visited_clusters, clusters_coverage