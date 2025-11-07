# import gurobipy as gp
# from gurobipy import GRB
# import networkx as nx
# import numpy as np
# from datagen import generate_sparse_network, generate_quantum_demands, print_network

# class QuantumNetworkRouter:
#     def __init__(self, graph, cost_matrix, demand_matrix):
#         """
#         graph: NetworkX graph or adjacency dict
#         cost_matrix: dict {(i,j): cost} for each edge
#         demand_matrix: list of dicts [{'src': 'a', 'dst': 'b', 'budget': 10, 'epr_pairs': 5}]
#         """
#         self.graph = graph
#         self.cost_matrix = cost_matrix
#         self.demands = demand_matrix
#         self.nodes = list(graph.nodes()) if hasattr(graph, 'nodes') else list(graph.keys())
#         self.edges = list(cost_matrix.keys())
        
#     def solve_ilp(self, time_limit=300, mip_gap=0.01):
#         """Solve the ILP using Gurobi"""
        
#         # Create model
#         model = gp.Model("QuantumNetworkRouting")
#         model.setParam('TimeLimit', time_limit)
#         model.setParam('MIPGap', mip_gap)
#         model.setParam('Threads', 4)  # Use 4 cores
        
#         # Decision variables
#         # x[k]: 1 if demand k is satisfied
#         x = model.addVars(len(self.demands), vtype=GRB.BINARY, name="x")
        
#         # f[i,j,k]: 1 if edge (i,j) is used for demand k
#         f = model.addVars(
#             [(i, j, k) for i, j in self.edges for k in range(len(self.demands))],
#             vtype=GRB.BINARY, 
#             name="f"
#         )
        
#         print(f"Created {len(x)} satisfaction variables")
#         print(f"Created {len(f)} flow variables")
#         print(f"Total variables: {len(x) + len(f)}")
        
#         # Objective: Maximize budget-weighted satisfaction
#         model.setObjective(
#             gp.quicksum(self.demands[k]['budget'] * x[k] for k in range(len(self.demands))),
#             GRB.MAXIMIZE
#         )
        
#         # Constraints
#         self._add_flow_conservation_constraints(model, x, f)
#         self._add_budget_constraints(model, x, f)
#         self._add_edge_disjointness_constraints(model, f)
        
#         # Solve
#         print("Starting optimization...")
#         model.optimize()
        
#         # Extract solution
#         if model.status == GRB.OPTIMAL:
#             return self._extract_solution(model, x, f)
#         else:
#             print(f"Optimization ended with status: {model.status}")
#             return None
    
#     def _add_flow_conservation_constraints(self, model, x, f):
#         """Add flow conservation constraints for each demand and node"""
#         for k, demand in enumerate(self.demands):
#             src, dst = demand['src'], demand['dst']
            
#             for node in self.nodes:
#                 # Calculate in-flow and out-flow for this node
#                 inflow = gp.quicksum(
#                     f[i, j, k] for i, j in self.edges 
#                     if j == node and (i, j) in self.edges
#                 )
#                 outflow = gp.quicksum(
#                     f[i, j, k] for i, j in self.edges 
#                     if i == node and (i, j) in self.edges
#                 )
                
#                 # Flow conservation based on node type
#                 if node == src:
#                     model.addConstr(outflow - inflow == x[k], f"flow_conservation_{k}_{node}")
#                 elif node == dst:
#                     model.addConstr(outflow - inflow == -x[k], f"flow_conservation_{k}_{node}")
#                 else:
#                     model.addConstr(outflow - inflow == 0, f"flow_conservation_{k}_{node}")
    
#     def _add_budget_constraints(self, model, x, f):
#         """Add budget constraints for each demand"""
#         for k, demand in enumerate(self.demands):
#             budget = demand['budget']
#             total_cost = gp.quicksum(
#                 self.cost_matrix[i, j] * f[i, j, k] 
#                 for i, j in self.edges
#             )
#             model.addConstr(total_cost <= budget * x[k], f"budget_{k}")
    
#     def _add_edge_disjointness_constraints(self, model, f):
#         """Add edge disjointness constraints"""
#         for i, j in self.edges:
#             model.addConstr(
#                 gp.quicksum(f[i, j, k] for k in range(len(self.demands))) <= 1,
#                 f"edge_disjoint_{i}_{j}"
#             )
    
#     def _extract_solution(self, model, x, f):
#         """Extract the solution from the model"""
#         solution = {
#             'objective_value': model.objVal,
#             'satisfied_demands': [],
#             'paths': {},
#             'total_demands': len(self.demands),
#             'satisfied_count': 0
#         }
        
#         for k in range(len(self.demands)):
#             if x[k].x > 0.5:  # Binary variable is 1
#                 demand = self.demands[k]
#                 solution['satisfied_demands'].append(k)
#                 solution['satisfied_count'] += 1
                
#                 # Reconstruct path
#                 path = self._reconstruct_path(f, k, demand['src'], demand['dst'])
#                 solution['paths'][k] = {
#                     'src': demand['src'],
#                     'dst': demand['dst'],
#                     'path': path,
#                     'budget': demand['budget'],
#                     'cost': sum(self.cost_matrix[path[i], path[i+1]] for i in range(len(path)-1))
#                 }
        
#         return solution
    
#     def _reconstruct_path(self, f, k, src, dst):
#         """Reconstruct the path for demand k from flow variables"""
#         path = [src]
#         current = src
        
#         while current != dst:
#             for i, j in self.edges:
#                 if i == current and f[i, j, k].x > 0.5:
#                     path.append(j)
#                     current = j
#                     break
        
#         return path

# # Example Usage
# if __name__ == "__main__":
#     # Define network
#     # graph = {
#     #     'a': ['b', 'd'],
#     #     'b': ['a','c','f'],
#     #     'c': ['b', 'e', 'd'],
#     #     'd': ['a', 'c', 'e'],
#     #     'e': ['c', 'd','f'],
#     #     'f': ['b','e']
#     # }
    
#     # # Cost matrix (symmetric)
#     # cost_matrix = {
#     #     ('a', 'b'): 3, ('b', 'a'): 3,
#     #     ('a', 'd'): 5, ('d', 'a'): 5,
#     #     ('b', 'c'): 3, ('c', 'b'): 3,
#     #     ('c', 'e'): 1, ('e', 'c'): 1,
#     #     ('c', 'd'): 1, ('d', 'c'): 1,
#     #     ('d', 'e'): 2, ('e', 'd'): 2,
#     #     ('b', 'f'): 2, ('f', 'b'): 2,
#     #     ('e', 'f'): 3, ('f', 'e'): 3
#     # }
    
#     # # Demand matrix
#     # demand_matrix = [
#     #     {'src': 'a', 'dst': 'e', 'epr_pairs': 5, 'budget': 7},
#     #     {'src': 'd', 'dst': 'f', 'epr_pairs': 3, 'budget': 6},
#     #     # {'src': 'e', 'dst': 'a', 'epr_pairs': 2, 'budget': 6}
#     # ]

#     # Generate network
#     graph, cost_matrix = generate_sparse_network(
#         num_nodes=5,
#         target_edges=8,
#         topology_type='random_geometric'  # or 'scale_free', 'small_world', 'hierarchical'
#     )

#     # Generate demands
#     demand_matrix = generate_quantum_demands(
#         nodes=list(graph.keys()),
#         num_demands=3,
#         demand_distribution='uniform'
#     )

#     # Get formatted output for your ILP code
#     network_output = print_network(graph, cost_matrix, demand_matrix)
#     print(network_output)

#     # Solve
#     router = QuantumNetworkRouter(graph, cost_matrix, demand_matrix)
#     solution = router.solve_ilp(time_limit=60)
    
#     if solution:
#         print(f"\nOptimal solution found!")
#         print(f"Objective value: {solution['objective_value']}")
#         print(f"Satisfied demands: {solution['satisfied_count']}/{solution['total_demands']}")
        
#         for k in solution['satisfied_demands']:
#             path_info = solution['paths'][k]
#             print(f"Demand {k}: {path_info['src']} -> {path_info['dst']}")
#             print(f"  Path: {' -> '.join(path_info['path'])}")
#             print(f"  Cost: {path_info['cost']}/{path_info['budget']}")
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np
import os
from datetime import datetime
from datagen import generate_sparse_network, generate_quantum_demands

# ============ LOGGING FUNCTION (TXT FORMAT) ============
def log_run(graph, cost_matrix, demand_matrix, solution, log_folder=r'C:\Users\babur\OneDrive\Desktop\Quantum_Research_Papers\Project-MVP Sir\Gurobi_Logs'):
    """
    Log the input (graph, cost_matrix, demand_matrix) and output (solution) to a .txt file
    in the date_logs folder with timestamp as filename.
    """
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    timestamp = datetime.now().strftime('%Y-%m-%d')
    filename = f"log_{timestamp}.txt"
    filepath = os.path.join(log_folder, filename)
    
    # Convert graph to serializable format
    graph_dict = dict(graph) if isinstance(graph, dict) else nx.to_dict_of_lists(graph)
    
    # Write to TXT file
    with open(filepath, 'a') as f:
        f.write("="*80 + "\n")
        f.write(f"QUANTUM NETWORK ROUTING - EXECUTION LOG\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        # ===== INPUT SECTION =====
        f.write("INPUT PARAMETERS\n")
        f.write("-"*80 + "\n\n")
        
        # Graph
        f.write("1. NETWORK GRAPH (Adjacency List)\n")
        f.write("-"*40 + "\n")
        for node, neighbors in sorted(graph_dict.items()):
            f.write(f"   Node '{node}': {neighbors}\n")
        f.write(f"\n")
        
        # Cost Matrix
        f.write("2. COST MATRIX (Edge Costs)\n")
        f.write("-"*40 + "\n")
        for (src, dst), cost in sorted(cost_matrix.items()):
            f.write(f"   ({src}, {dst}): {cost}\n")
        f.write(f"\n")
        
        # Demand Matrix
        f.write("3. DEMAND MATRIX (Routing Requests)\n")
        f.write("-"*40 + "\n")
        for i, demand in enumerate(demand_matrix):
            f.write(f"   Demand {i}:\n")
            f.write(f"      Source: {demand['src']}\n")
            f.write(f"      Destination: {demand['dst']}\n")
            f.write(f"      Budget: {demand['budget']}\n")
            f.write(f"      EPR Pairs Required: {demand.get('epr_pairs', 'N/A')}\n\n")
        
        # ===== OUTPUT/SOLUTION SECTION =====
        f.write("\n" + "="*80 + "\n")
        f.write("SOLUTION & RESULTS\n")
        f.write("="*80 + "\n\n")
        
        if solution and not solution.get('failed', False):
            f.write("STATUS:  SOLUTION FOUND\n\n")
            
            f.write("SUMMARY METRICS\n")
            f.write("-"*40 + "\n")
            f.write(f"   Objective Value: {solution.get('objective_value', 'N/A')}\n")
            f.write(f"   Satisfied Demands: {solution.get('satisfied_count', 0)} / {solution.get('total_demands', 0)}\n")
            f.write(f"   Total Cost: {solution.get('total_cost', 0)}\n")
            f.write(f"   Total Hops: {solution.get('total_hops', 0)}\n\n")
            
            f.write("SATISFIED DEMAND DETAILS\n")
            f.write("-"*40 + "\n")
            
            if solution.get('satisfied_demands'):
                for k in solution['satisfied_demands']:
                    path_info = solution['paths'].get(k, {})
                    f.write(f"\n   Demand {k}:\n")
                    f.write(f"      Source: {path_info.get('src', 'N/A')}\n")
                    f.write(f"      Destination: {path_info.get('dst', 'N/A')}\n")
                    f.write(f"      Path: {' -> '.join(path_info.get('path', []))}\n")
                    f.write(f"      Cost Used: {path_info.get('cost', 0)} / {path_info.get('budget', 0)}\n")
                    f.write(f"      Number of Hops: {path_info.get('hops', 0)}\n")
            else:
                f.write("   No demands satisfied.\n")
            
            # Unsatisfied demands
            unsatisfied = [i for i in range(solution.get('total_demands', 0)) 
                          if i not in solution.get('satisfied_demands', [])]
            if unsatisfied:
                f.write(f"\n\n   UNSATISFIED DEMANDS: {unsatisfied}\n")
        
        else:
            f.write("STATUS: ✗ SOLUTION FAILED\n\n")
            if solution:
                f.write(f"   Reason: {solution.get('status', 'Unknown')}\n")
            else:
                f.write(f"   Reason: No solution returned\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF LOG\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Log saved to {filepath}")


# ============ QUANTUM NETWORK ROUTER CLASS ============
class QuantumNetworkRouter:
    def __init__(self, graph, cost_matrix, demand_matrix):
        """
        graph: NetworkX graph or adjacency dict
        cost_matrix: dict {(i,j): cost} for each edge
        demand_matrix: list of dicts [{'src': 'a', 'dst': 'b', 'budget': 10, 'epr_pairs': 5}]
        """
        self.graph = graph
        self.cost_matrix = cost_matrix
        self.demands = demand_matrix
        self.nodes = list(graph.nodes()) if hasattr(graph, 'nodes') else list(graph.keys())
        self.edges = list(cost_matrix.keys())

    def solve_ilp(self, time_limit=300, mip_gap=0.01):
        """Solve the ILP using Gurobi"""
        # Create model
        model = gp.Model("QuantumNetworkRouting")
        model.setParam('TimeLimit', time_limit)
        model.setParam('MIPGap', mip_gap)
        model.setParam('Threads', 4)  # Use 4 cores
        model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        # Decision variables
        # x[k]: 1 if demand k is satisfied
        x = model.addVars(len(self.demands), vtype=GRB.BINARY, name="x")
        
        # f[i,j,k]: 1 if edge (i,j) is used for demand k
        f = model.addVars(
            [(i, j, k) for i, j in self.edges for k in range(len(self.demands))],
            vtype=GRB.BINARY,
            name="f"
        )

        print(f"Created {len(x)} satisfaction variables")
        print(f"Created {len(f)} flow variables")
        print(f"Total variables: {len(x) + len(f)}")

        # Objective: Maximize budget-weighted satisfaction
        model.setObjective(
            gp.quicksum(self.demands[k]['budget'] * x[k] for k in range(len(self.demands))),
            GRB.MAXIMIZE
        )

        # Constraints
        self._add_flow_conservation_constraints(model, x, f)
        self._add_budget_constraints(model, x, f)
        self._add_edge_disjointness_constraints(model, f)

        # Solve
        print("Starting optimization...")
        model.optimize()

        # Extract solution and log
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            solution = self._extract_solution(model, x, f)
            
            # Log the run
            log_run(self.graph, self.cost_matrix, self.demands, solution)
            
            return solution
        else:
            print(f"Optimization ended with status: {model.status}")
            failed_solution = {"status": f"Optimization Status {model.status}", "failed": True}
            
            # Log the failed run
            log_run(self.graph, self.cost_matrix, self.demands, failed_solution)
            
            return None

    def _add_flow_conservation_constraints(self, model, x, f):
        """Add flow conservation constraints for each demand and node"""
        for k, demand in enumerate(self.demands):
            src, dst = demand['src'], demand['dst']
            
            for node in self.nodes:
                # Calculate in-flow and out-flow for this node
                inflow = gp.quicksum(
                    f[i, j, k] for i, j in self.edges
                    if j == node and (i, j) in self.edges
                )
                outflow = gp.quicksum(
                    f[i, j, k] for i, j in self.edges
                    if i == node and (i, j) in self.edges
                )
                
                # Flow conservation based on node type
                if node == src:
                    model.addConstr(outflow - inflow == x[k], f"flow_conservation_{k}_{node}")
                elif node == dst:
                    model.addConstr(outflow - inflow == -x[k], f"flow_conservation_{k}_{node}")
                else:
                    model.addConstr(outflow - inflow == 0, f"flow_conservation_{k}_{node}")

    def _add_budget_constraints(self, model, x, f):
        """Add budget constraints for each demand"""
        for k, demand in enumerate(self.demands):
            budget = demand['budget']
            total_cost = gp.quicksum(
                self.cost_matrix[i, j] * f[i, j, k]
                for i, j in self.edges
            )
            model.addConstr(total_cost <= budget * x[k], f"budget_{k}")

    def _add_edge_disjointness_constraints(self, model, f):
        """Add edge disjointness constraints"""
        for i, j in self.edges:
            model.addConstr(
                gp.quicksum(f[i, j, k] for k in range(len(self.demands))) <= 1,
                f"edge_disjoint_{i}_{j}"
            )

    def _extract_solution(self, model, x, f):
        """Extract the solution from the model"""
        solution = {
            'objective_value': model.objVal,
            'satisfied_demands': [],
            'paths': {},
            'total_demands': len(self.demands),
            'satisfied_count': 0,
            'total_cost': 0,
            'total_hops': 0
        }

        for k in range(len(self.demands)):
            if x[k].x > 0.5:  # Binary variable is 1
                demand = self.demands[k]
                solution['satisfied_demands'].append(k)
                solution['satisfied_count'] += 1
                
                # Reconstruct path
                path = self._reconstruct_path(f, k, demand['src'], demand['dst'])
                cost = sum(self.cost_matrix[path[i], path[i+1]] for i in range(len(path)-1))
                hops = len(path) - 1
                
                solution['paths'][k] = {
                    'src': demand['src'],
                    'dst': demand['dst'],
                    'path': path,
                    'budget': demand['budget'],
                    'cost': cost,
                    'hops': hops
                }
                
                solution['total_cost'] += cost
                solution['total_hops'] += hops

        return solution

    def _reconstruct_path(self, f, k, src, dst):
        """Reconstruct the path for demand k from flow variables"""
        path = [src]
        current = src

        while current != dst:
            for i, j in self.edges:
                if i == current and f[i, j, k].x > 0.5:
                    path.append(j)
                    current = j
                    break

        return path


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Define network
    # graph = {
    #     'a': ['b', 'd'],
    #     'b': ['a', 'c', 'f'],
    #     'c': ['b', 'e', 'd'],
    #     'd': ['a', 'c', 'e'],
    #     'e': ['c', 'd', 'f'],
    #     'f': ['b', 'e']
    # }
    
    # # Cost matrix (symmetric)
    # cost_matrix = {
    #     ('a', 'b'): 3, ('b', 'a'): 3,
    #     ('a', 'd'): 5, ('d', 'a'): 5,
    #     ('b', 'c'): 3, ('c', 'b'): 3,
    #     ('c', 'e'): 1, ('e', 'c'): 1,
    #     ('c', 'd'): 1, ('d', 'c'): 1,
    #     ('d', 'e'): 2, ('e', 'd'): 2,
    #     ('b', 'f'): 2, ('f', 'b'): 2,
    #     ('e', 'f'): 3, ('f', 'e'): 3
    # }
    
    # # Demand matrix
    # demand_matrix = [
    #     {'src': 'a', 'dst': 'e', 'epr_pairs': 5, 'budget': 7},
    #     {'src': 'd', 'dst': 'f', 'epr_pairs': 3, 'budget': 6},
    # ]

    # Generate network
    graph, cost_matrix = generate_sparse_network(
        num_nodes=5,
        target_edges=6,
        topology_type='random_geometric'
    )

    # Generate demands (only between non-adjacent nodes)
    demand_matrix, actual_num, max_possible = generate_quantum_demands(
        graph=graph,
        num_demands=3,
        demand_distribution='uniform'
    )

    print(f"Requested: num demands")
    print(f"Generated: {actual_num} demands")
    print(f"Maximum possible: {max_possible} non-adjacent pairs")
    
    # Solve
    router = QuantumNetworkRouter(graph, cost_matrix, demand_matrix)
    solution = router.solve_ilp(time_limit=60)
    
    if solution:
        print(f"\n✓ Optimal solution found!")
        print(f"Objective value: {solution['objective_value']}")
        print(f"Satisfied demands: {solution['satisfied_count']}/{solution['total_demands']}")
        print(f"Total cost: {solution['total_cost']}")
        print(f"Total hops: {solution['total_hops']}\n")
        
        for k in solution['satisfied_demands']:
            path_info = solution['paths'][k]
            print(f"Demand {k}: {path_info['src']} -> {path_info['dst']}")
            print(f"  Path: {' -> '.join(path_info['path'])}")
            print(f"  Cost: {path_info['cost']}/{path_info['budget']} | Hops: {path_info['hops']}")
    else:
        print("\n✗ No solution found!")
