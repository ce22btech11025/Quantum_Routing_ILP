# Quantum Network Data Generator
# Generates sparse, realistic network topologies for quantum network ILP problems

import random
import math

def generate_sparse_network(num_nodes=20, target_edges=1000, topology_type='random_geometric'):
    """
    Generate a sparse network topology suitable for quantum networks
    
    Parameters:
    - num_nodes: number of nodes
    - target_edges: target number of edges
    - topology_type: 'random_geometric', 'scale_free', 'small_world', or 'hierarchical'
    """
    
    nodes = [chr(97 + i) if i < 26 else f'n{i}' for i in range(num_nodes)]
    
    if topology_type == 'random_geometric':
        return _generate_random_geometric(nodes, target_edges)
    elif topology_type == 'scale_free':
        return _generate_scale_free(nodes, target_edges)
    elif topology_type == 'small_world':
        return _generate_small_world(nodes, target_edges)
    elif topology_type == 'hierarchical':
        return _generate_hierarchical(nodes, target_edges)
    else:
        return _generate_random_geometric(nodes, target_edges)


def _generate_random_geometric(nodes, target_edges):
    """
    Generate random geometric graph where nodes are placed in 2D space
    and edges connect nearby nodes (realistic for real networks)
    """
    random.seed(42)
    num_nodes = len(nodes)
    
    # Place nodes randomly in a unit square
    positions = {node: (random.random(), random.random()) for node in nodes}
    
    # Connect nodes based on distance threshold
    graph = {node: [] for node in nodes}
    cost_matrix = {}
    edges = set()
    
    # Start with distance-based connectivity
    edges_list = []
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i < j:
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]
                dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                edges_list.append((dist, node1, node2))
    
    # Sort by distance and connect nearby nodes
    edges_list.sort()
    
    for dist, node1, node2 in edges_list:
        if len(edges) >= target_edges:
            break
        
        # Add edge with distance-based cost
        edge = tuple(sorted([node1, node2]))
        if edge not in edges:
            edges.add(edge)
            graph[node1].append(node2)
            graph[node2].append(node1)
            
            # Cost inversely proportional to distance + random factor
            cost = max(1, int(15 * dist) + random.randint(-2, 2))
            cost_matrix[(node1, node2)] = cost
            cost_matrix[(node2, node1)] = cost
    
    # If not enough edges, add random ones
    while len(edges) < target_edges:
        node1, node2 = random.sample(nodes, 2)
        edge = tuple(sorted([node1, node2]))
        
        if edge not in edges:
            edges.add(edge)
            graph[node1].append(node2)
            graph[node2].append(node1)
            
            cost = random.randint(1, 15)
            cost_matrix[(node1, node2)] = cost
            cost_matrix[(node2, node1)] = cost
    
    # Sort adjacency lists
    for node in graph:
        graph[node] = sorted(list(set(graph[node])))
    
    return graph, cost_matrix


def _generate_scale_free(nodes, target_edges):
    """
    Generate scale-free network (Preferential Attachment)
    Mimics real network topologies where some nodes have many connections
    """
    random.seed(42)
    num_nodes = len(nodes)
    
    graph = {node: [] for node in nodes}
    cost_matrix = {}
    edges = set()
    
    # Start with a complete graph of first 3 nodes
    initial_nodes = nodes[:3]
    for i, n1 in enumerate(initial_nodes):
        for n2 in initial_nodes[i+1:]:
            graph[n1].append(n2)
            graph[n2].append(n1)
            edges.add(tuple(sorted([n1, n2])))
            cost = random.randint(1, 15)
            cost_matrix[(n1, n2)] = cost
            cost_matrix[(n2, n1)] = cost
    
    # Add remaining nodes with preferential attachment
    for new_node in nodes[3:]:
        # Connect to existing nodes with probability proportional to degree
        degrees = [len(graph[n]) for n in nodes if n != new_node]
        total_degree = sum(degrees)
        
        # Add 3-5 connections per new node
        connections = random.randint(3, min(5, len(nodes) - 1))
        
        for _ in range(connections):
            if len(edges) >= target_edges:
                break
            
            # Pick node with probability proportional to degree
            probabilities = [d / total_degree for d in degrees]
            existing_nodes = [n for n in nodes if n != new_node]
            chosen = random.choices(existing_nodes, weights=probabilities, k=1)[0]
            
            edge = tuple(sorted([new_node, chosen]))
            if edge not in edges:
                edges.add(edge)
                graph[new_node].append(chosen)
                graph[chosen].append(new_node)
                
                cost = random.randint(1, 15)
                cost_matrix[(new_node, chosen)] = cost
                cost_matrix[(chosen, new_node)] = cost
    
    # Add more random edges if needed
    while len(edges) < target_edges:
        node1, node2 = random.sample(nodes, 2)
        edge = tuple(sorted([node1, node2]))
        
        if edge not in edges:
            edges.add(edge)
            graph[node1].append(node2)
            graph[node2].append(node1)
            
            cost = random.randint(1, 15)
            cost_matrix[(node1, node2)] = cost
            cost_matrix[(node2, node1)] = cost
    
    # Sort adjacency lists
    for node in graph:
        graph[node] = sorted(list(set(graph[node])))
    
    return graph, cost_matrix


def _generate_small_world(nodes, target_edges):
    """
    Generate small-world network (Watts-Strogatz style)
    Combines local clustering with long-range shortcuts
    """
    random.seed(42)
    num_nodes = len(nodes)
    
    graph = {node: [] for node in nodes}
    cost_matrix = {}
    edges = set()
    
    # Create ring topology with local connections
    k = 4  # Each node connects to k nearest neighbors
    for i, node in enumerate(nodes):
        for offset in range(1, k // 2 + 1):
            neighbor = nodes[(i + offset) % num_nodes]
            edge = tuple(sorted([node, neighbor]))
            
            if edge not in edges:
                edges.add(edge)
                graph[node].append(neighbor)
                graph[neighbor].append(node)
                cost = random.randint(1, 8)
                cost_matrix[(node, neighbor)] = cost
                cost_matrix[(neighbor, node)] = cost
    
    # Add random long-range shortcuts
    while len(edges) < target_edges:
        node1, node2 = random.sample(nodes, 2)
        edge = tuple(sorted([node1, node2]))
        
        if edge not in edges:
            edges.add(edge)
            graph[node1].append(node2)
            graph[node2].append(node1)
            
            cost = random.randint(5, 15)
            cost_matrix[(node1, node2)] = cost
            cost_matrix[(node2, node1)] = cost
    
    # Sort adjacency lists
    for node in graph:
        graph[node] = sorted(list(set(graph[node])))
    
    return graph, cost_matrix


def _generate_hierarchical(nodes, target_edges):
    """
    Generate hierarchical network topology
    Suitable for metropolitan/regional quantum networks
    """
    random.seed(42)
    num_nodes = len(nodes)
    
    graph = {node: [] for node in nodes}
    cost_matrix = {}
    edges = set()
    
    # Divide nodes into hierarchical levels
    level1_size = max(2, num_nodes // 4)
    level2_size = num_nodes - level1_size
    
    level1_nodes = nodes[:level1_size]
    level2_nodes = nodes[level1_size:]
    
    # Connect within level 1 (backbone)
    for i, n1 in enumerate(level1_nodes):
        for n2 in level1_nodes[i+1:]:
            edge = tuple(sorted([n1, n2]))
            edges.add(edge)
            graph[n1].append(n2)
            graph[n2].append(n1)
            cost = random.randint(1, 8)
            cost_matrix[(n1, n2)] = cost
            cost_matrix[(n2, n1)] = cost
    
    # Connect level 2 to level 1
    for node2 in level2_nodes:
        # Each level 2 node connects to 1-2 level 1 nodes
        connections = random.randint(1, min(2, len(level1_nodes)))
        for _ in range(connections):
            node1 = random.choice(level1_nodes)
            edge = tuple(sorted([node1, node2]))
            
            if edge not in edges:
                edges.add(edge)
                graph[node1].append(node2)
                graph[node2].append(node1)
                cost = random.randint(5, 15)
                cost_matrix[(node1, node2)] = cost
                cost_matrix[(node2, node1)] = cost
    
    # Add connections within level 2 for local clustering
    while len(edges) < target_edges:
        node1, node2 = random.sample(level2_nodes, 2)
        edge = tuple(sorted([node1, node2]))
        
        if edge not in edges:
            edges.add(edge)
            graph[node1].append(node2)
            graph[node2].append(node1)
            cost = random.randint(3, 12)
            cost_matrix[(node1, node2)] = cost
            cost_matrix[(node2, node1)] = cost
    
    # Sort adjacency lists
    for node in graph:
        graph[node] = sorted(list(set(graph[node])))
    
    return graph, cost_matrix


def generate_quantum_demands(graph, num_demands=30, demand_distribution='uniform'):
    """
    Generate quantum network demands where src and dst are NOT directly connected by an edge
    
    Parameters:
    - graph: adjacency dictionary of the network
    - num_demands: number of demands to generate
    - demand_distribution: 'uniform' or 'skewed'
    
    Returns:
    - demand_matrix: list of demand dictionaries
    - actual_num_demands: actual number of demands generated (may be less than requested)
    - max_possible_demands: maximum possible non-adjacent demand pairs
    """
    random.seed(10)
    nodes = list(graph.keys())
    
    # Build set of connected pairs (edges)
    connected_pairs = set()
    for u in graph:
        for v in graph[u]:
            # Normalize to avoid duplicates (smaller, larger)
            pair = tuple(sorted([u, v]))
            connected_pairs.add(pair)
    
    # Find all non-adjacent pairs (valid demand pairs)
    valid_demand_pairs = []
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            pair = tuple(sorted([node1, node2]))
            # Check if this pair is NOT in connected_pairs
            if pair not in connected_pairs:
                valid_demand_pairs.append((node1, node2))
                # Also add reverse direction
                valid_demand_pairs.append((node2, node1))
    
    max_possible_demands = len(valid_demand_pairs)
    
    # Limit to requested number of demands
    actual_num = min(num_demands, max_possible_demands)
    
    demand_matrix = []
    
    # Randomly sample from valid demand pairs
    if actual_num > 0:
        sampled_pairs = random.sample(valid_demand_pairs, actual_num)
        
        for src, dst in sampled_pairs:
            if demand_distribution == 'uniform':
                epr_pairs = random.randint(2, 10)
                budget = random.randint(15, 50)
            elif demand_distribution == 'skewed':
                # Some demands need many EPR pairs
                epr_pairs = random.choices(
                    [random.randint(1, 3), random.randint(4, 8), random.randint(8, 15)],
                    weights=[0.5, 0.3, 0.2]
                )[0]
                budget = random.randint(10, 60)
            else:
                epr_pairs = random.randint(2, 10)
                budget = random.randint(15, 50)
            
            demand_matrix.append({
                'src': src,
                'dst': dst,
                'epr_pairs': epr_pairs,
                'budget': budget
            })
    
    return demand_matrix, actual_num, max_possible_demands


def print_network(graph, cost_matrix, demand_matrix, output_file=None):
    """Pretty print the network in the format you need"""
    
    output = []
    output.append("# Define network")
    output.append("graph = {")
    for node in sorted(graph.keys()):
        output.append(f"    '{node}': {graph[node]},")
    output.append("}")
    
    output.append("\n# Cost matrix (symmetric)")
    output.append("cost_matrix = {")
    
    # Get unique edges
    edges = set()
    for u in graph:
        for v in graph[u]:
            edge = tuple(sorted([u, v]))
            edges.add(edge)
    
    for u, v in sorted(edges):
        output.append(f"    ('{u}', '{v}'): {cost_matrix[(u, v)]}, ('{v}', '{u}'): {cost_matrix[(v, u)]},")
    output.append("}")
    
    output.append("\n# Demand matrix")
    output.append("demand_matrix = [")
    for demand in demand_matrix:
        output.append(f"    {demand},")
    output.append("]")
    
    result = "\n".join(output)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(result)
    
    return result


# ============================================================================
# MAIN: Generate networks for your ILP problem
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("QUANTUM NETWORK DATA GENERATOR")
    print("=" * 80)
    
    # Generate network using different topologies
    topologies = ['random_geometric', 'scale_free', 'small_world', 'hierarchical']
    
    for topology in topologies:
        print(f"\n\n{'=' * 80}")
        print(f"Generating {topology.upper()} Network")
        print('=' * 80)
        
        graph, cost_matrix = generate_sparse_network(
            num_nodes=20,
            target_edges=1000,
            topology_type=topology
        )
        
        # Generate demands - only between non-adjacent nodes
        demands, actual_demands, max_possible = generate_quantum_demands(
            graph=graph,
            num_demands=30,
            demand_distribution='uniform'
        )
        
        # Statistics
        total_edges = len(set(tuple(sorted([u, v])) for u in graph for v in graph[u]))
        avg_degree = sum(len(neighbors) for neighbors in graph.values()) / len(graph)
        
        print(f"\nNetwork Statistics:")
        print(f"  Nodes: {len(graph)}")
        print(f"  Edges: {total_edges}")
        print(f"  Average Degree: {avg_degree:.2f}")
        print(f"  Maximum possible non-adjacent demand pairs: {max_possible}")
        print(f"  Demands requested: 30")
        print(f"  Demands generated: {actual_demands}")
        
        if actual_demands < 30:
            print(f"  ⚠️  Note: Only {actual_demands} non-adjacent pairs available (less than requested 30)")
            print(f"  ⚠️  These are the MAXIMUM POSSIBLE demand pairs for this network")
        
        # Print first topology in detail
        if topology == 'random_geometric':
            network_output = print_network(graph, cost_matrix, demands)
            print(f"\n{network_output[:500]}...\n[Output truncated for display]")
            
            # Save to file
            print_network(graph, cost_matrix, demands, 
                         output_file=f'quantum_network_{topology}.py')
            print(f"\n✓ Saved to: quantum_network_{topology}.py")