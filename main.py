import gurobipy as gp
from gurobipy import GRB
import networkx as nx
import numpy as np

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
        
        # Extract solution
        if model.status == GRB.OPTIMAL:
            return self._extract_solution(model, x, f)
        else:
            print(f"Optimization ended with status: {model.status}")
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
            'satisfied_count': 0
        }
        
        for k in range(len(self.demands)):
            if x[k].x > 0.5:  # Binary variable is 1
                demand = self.demands[k]
                solution['satisfied_demands'].append(k)
                solution['satisfied_count'] += 1
                
                # Reconstruct path
                path = self._reconstruct_path(f, k, demand['src'], demand['dst'])
                solution['paths'][k] = {
                    'src': demand['src'],
                    'dst': demand['dst'],
                    'path': path,
                    'budget': demand['budget'],
                    'cost': sum(self.cost_matrix[path[i], path[i+1]] for i in range(len(path)-1))
                }
        
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

# Example Usage
if __name__ == "__main__":
    # Define network
    graph = {
        'a': ['b', 'd'],
        'b': ['a','c'],
        'c': ['b', 'e', 'd'],
        'd': ['a', 'c', 'e'],
        'e': ['c', 'd'],
    }
    
    # Cost matrix (symmetric)
    cost_matrix = {
        ('a', 'b'): 2, ('b', 'a'): 2,
        ('a', 'd'): 5, ('d', 'a'): 5,
        ('b', 'c'): 3, ('c', 'b'): 3,
    }
    
    # Demand matrix
    demand_matrix = [
        {'src': 'a', 'dst': 'e', 'epr_pairs': 5, 'budget': 5},
        # {'src': 'c', 'dst': 'd', 'epr_pairs': 3, 'budget': 4},
        # {'src': 'e', 'dst': 'a', 'epr_pairs': 2, 'budget': 6}
    ]
    
    # Solve
    router = QuantumNetworkRouter(graph, cost_matrix, demand_matrix)
    solution = router.solve_ilp(time_limit=60)
    
    if solution:
        print(f"\nOptimal solution found!")
        print(f"Objective value: {solution['objective_value']}")
        print(f"Satisfied demands: {solution['satisfied_count']}/{solution['total_demands']}")
        
        for k in solution['satisfied_demands']:
            path_info = solution['paths'][k]
            print(f"Demand {k}: {path_info['src']} -> {path_info['dst']}")
            print(f"  Path: {' -> '.join(path_info['path'])}")
            print(f"  Cost: {path_info['cost']}/{path_info['budget']}")
