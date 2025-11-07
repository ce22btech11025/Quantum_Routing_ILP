# from dotenv import load_dotenv
# import os

# load_dotenv()  # loads environment variables from .env file

# import gurobipy as gp

# try:
#     # Attempt to create a Gurobi environment
#     env = gp.Env(empty=True)
#     env.start()
#     print("Gurobi is installed and licensed.")
#     env.dispose()
# except gp.GurobiError as e:
#     print(f"Gurobi error: {e}")
#     print("Gurobi may not be installed or licensed correctly.")
# # import gurobipy
# # print(gurobipy.gurobi.version())

# from gurobipy import Model, GRB

# # Create model
# model = Model("small_ilp_test")

# # Set log file in same directory
# model.setParam("LogFile", "gurobi_run.log")

# # Variables (integer)
# x = model.addVar(vtype=GRB.INTEGER, name="x")
# y = model.addVar(vtype=GRB.INTEGER, name="y")

# # Objective: maximize x + 2y
# model.setObjective(x + 2*y, GRB.MAXIMIZE)

# # Constraints
# model.addConstr(x + y <= 4, "c1")
# model.addConstr(x - y >= 1, "c2")

# # Optimize
# model.optimize()

# # Print results
# if model.status == GRB.OPTIMAL:
#     print("\nOptimal Solution:")
#     print(f"x = {x.X}")
#     print(f"y = {y.X}")
#     print(f"Objective value = {model.ObjVal}")
# else:
#     print("No optimal solution found.")

import random
from gurobipy import Model, GRB

# Problem dimensions
n = 1000

# Generate random weights and values
random.seed(42)
weights = [random.randint(1, 100) for _ in range(n)]
values = [random.randint(1, 50) for _ in range(n)]

# Capacity (around 25% of total weight)
capacity = sum(weights) * 0.25

# Create model
model = Model("1000_var_knapsack")
model.setParam("LogFile", "Gurobi_Logs\gurobi_1000_vars.log")

# Decision variables (binary)
x = model.addVars(n, vtype=GRB.BINARY, name="x")

# Objective: maximize total value
model.setObjective(sum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

# Constraint: total weight <= capacity
model.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity, "capacity")

# Optimize
model.optimize()

# Results
if model.status == GRB.OPTIMAL:
    solution_count = sum(int(x[i].X) for i in range(n))
    total_value = model.ObjVal
    print(f"✅ Optimal solution found")
    print(f"Selected items: {solution_count}/{n}")
    print(f"Total objective value: {total_value}")

else:
    print("❌ No optimal solution found")
