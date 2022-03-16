#Importing Libraries
import numpy as np
import pandas as pd
import random
import itertools
import time

#Importing Data
data = pd.ExcelFile('Soft_Computing_Data.xlsx')

#Importing tables individually
CAB_10_nodes_flow = pd.read_excel(data, '10_nodes_CAB_flow', header = None)  
CAB_10_nodes_cost = pd.read_excel(data, '10_nodes_CAB_cost', header = None)  
CAB_25_nodes_flow = pd.read_excel(data, '25_nodes_CAB_flow', header = None)    
CAB_25_nodes_cost = pd.read_excel(data, '25_nodes_CAB_cost', header = None)
TR_55_nodes_flow = pd.read_excel(data, '55_nodes_TR_flow', header = None)
TR_55_nodes_cost = pd.read_excel(data, '55_nodes_TR_cost', header = None)  
TR_81_nodes_flow = pd.read_excel(data, '81_nodes_TR_flow', header = None)    
TR_81_nodes_cost = pd.read_excel(data, '81_nodes_TR_cost', header = None)
RGP_100_nodes_flow = pd.read_excel(data, '100_nodes_RGP_flow', header = None)
RGP_100_nodes_cost = pd.read_excel(data, '100_nodes_RGP_cost', header = None)  
RGP_130_nodes_flow = pd.read_excel(data, '130_nodes_RGP_flow', header = None)    
RGP_130_nodes_cost = pd.read_excel(data, '130_nodes_RGP_cost', header = None)

#Calculating the network cost
def network_cost(initial_solution, flow_matrix, cost_matrix, alpha = 0.2):
    
    number_nodes = cost_matrix.shape[0]
    cost = 0
    flow = 0
    for node_1 in range(number_nodes):
        for node_2 in range(number_nodes):
            cost += flow_matrix[node_1][node_2] * (cost_matrix[node_1][initial_solution[node_1]-1] + 
                    alpha*cost_matrix[initial_solution[node_1] - 1][initial_solution[node_2] - 1] +
                    cost_matrix[initial_solution[node_2]-1][node_2]) 
            flow += flow_matrix[node_1][node_2]        
              
   
    return cost/flow

#For diversification
def neighbourhood_structure2(solution):
    neighbour_struc = solution.copy()
    hubs = list(set(neighbour_struc))
    nodes = list(range(1, len(neighbour_struc) + 1))
    spokes = list(set(nodes) - set(hubs))
    select_spoke = random.choice(spokes)
    hub = neighbour_struc[select_spoke - 1]
    select_hub = random.choice(hubs)
    while select_hub == hub:
        select_hub = random.choice(hubs)
    neighbour_struc[select_spoke - 1] = select_hub
    # print('Neighbourhood Structure2',neighbour_struc)
    return neighbour_struc

#Generated solution for particle positions
def solution_generator(Xparticle_1, Xparticle_2):
    array = [Xparticle_1, Xparticle_2]
    array_1 = array.copy()
    array_1[0] = [(sorted(array[0]).index(i)+1) for i in array[0]]
    array_1[1] = [int(i) for i in array[1]]
    hubs = array_1[0][:3]
    solution = [hubs[i-1] for i in array_1[1]]
    #repairing 
    for hub in hubs:
        solution[hub-1] = hub
        
    return solution

#Diversification 
def apply_diversification(neighbours):
    diversified_list = []
    for neighbour in neighbours:
        diversified_list.append(neighbour)
        neighbour_dup = neighbourhood_structure2(neighbour)
        if neighbour_dup not in diversified_list:
            diversified_list.append(neighbour_dup)
    return diversified_list

def particle_swarm_heuristic(flow_matrix, cost_matrix, n_hub, alpha):
  start = time.time()
  SwarmSize = 25
  node = flow_matrix.shape[0]
  Dimensions=node
  iter=0
  global_best_cost=10000
  Vmin = 0
  Vmax = n_hub * node * 0.1
  X_1_min = 0
  X_1_max = n_hub * node
  X_2_min = 1
  X_2_max = n_hub + 1 - 0.01
  particle_best_cost=10000
  n_iter = 1000
  XparticleSwarm=[]
  VparticleSwarm=[]
  for i in range(SwarmSize):
    X_particle_1 = []
    X_particle_2 = []
    V_particle_1 = []
    V_particle_2 = []
    for i in range(Dimensions):
      X_particle_1.append(X_1_max*random.uniform(0, 1))
      X_particle_2.append(random.uniform(X_2_min, X_2_max))
      V_particle_1.append(Vmin + (Vmax - Vmin) * random.uniform(0, 1))
      rand = random.uniform(0, 1)
      if rand>0.5:
        V_particle_2.append(Vmin + (Vmax - Vmin) * random.uniform(0, 1))
      else:
        V_particle_2.append(-1*(Vmin + (Vmax - Vmin) * random.uniform(0, 1)))
    
    XparticleSwarm.append([X_particle_1, X_particle_2])
    VparticleSwarm.append([V_particle_1, V_particle_2])
  
  while iter<n_iter:
    r1 = random.uniform(0, 1)
    r2 = random.uniform(0, 1)
    C1 = 2
    C2 = 2
    W = 1
    for i in range(SwarmSize):
      X_particle_1 = XparticleSwarm[i][0]
      X_particle_2 = XparticleSwarm[i][1]
      V_particle_1 = VparticleSwarm[i][0]
      V_particle_2 = VparticleSwarm[i][1]
      particle = solution_generator(X_particle_1, X_particle_2)
      particle_cost = network_cost(particle, flow_matrix, cost_matrix, alpha)

      if particle_cost < particle_best_cost:
          particle_best_cost = particle_cost
          particle_best_solution = particle
          local_best_particle = [X_particle_1, X_particle_2]

      if particle_cost < global_best_cost:
          global_best_cost = particle_cost
          global_best_solution = particle
          global_best_particle = [X_particle_1, X_particle_2]
    for i in range(SwarmSize-1):
      X_particle_1_next = XparticleSwarm[i + 1][0]
      X_particle_2_next = XparticleSwarm[i + 1][1]
      V_particle_1_next = VparticleSwarm[i + 1][0]
      V_particle_2_next = VparticleSwarm[i + 1][1]
      
      for d in range(Dimensions):
        V_particle_1_next[d] = W * V_particle_1[d] + C1 * r1 * (local_best_particle[0][d]- X_particle_1[d])\
                + C2 * r2 * (global_best_particle[0][d] - X_particle_1[d])
        V_particle_2_next[d] = W * V_particle_2[d] + C1 * r1 * (local_best_particle[1][d]- X_particle_2[d])\
                + C2 * r2 * (global_best_particle[1][d] - X_particle_2[d])
        
        if V_particle_1_next[d] > Vmax:
            V_particle_1_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
        if V_particle_1_next[d] < Vmin:
            V_particle_1_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
        
        if V_particle_2_next[d] > 0:
          if V_particle_2_next[d] > Vmax:
              V_particle_2_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
          if V_particle_2_next[d] < Vmin:
              V_particle_2_next[d] = Vmin + (Vmax - Vmin) * random.uniform(0, 1)
        else:
          if abs(V_particle_2_next[d]) > Vmax:
              V_particle_2_next[d] = -1*(Vmin + (Vmax - Vmin) * random.uniform(0, 1))
          if abs(V_particle_2_next[d]) < Vmin:
              V_particle_2_next[d] = -1*(Vmin + (Vmax - Vmin) * random.uniform(0, 1))

        X_particle_1_next[d] = X_particle_1[d] + V_particle_1_next[d]
        X_particle_2_next[d] = X_particle_2[d] + V_particle_2_next[d]

        if X_particle_1_next[d] > X_1_max:
          X_particle_1_next[d] = X_1_max*random.uniform(0, 1)
        if X_particle_1_next[d] < X_1_min:
          X_particle_1_next[d] = X_1_max*random.uniform(0, 1)

        if X_particle_2_next[d] > X_2_max:
          X_particle_2_next[d] = random.uniform(X_2_min, X_2_max)
        if X_particle_2_next[d] < X_2_min:
          X_particle_2_next[d] = random.uniform(X_2_min, X_2_max) 

        XparticleSwarm[i+1] = [X_particle_1_next, X_particle_2_next]
        VparticleSwarm[i+1] = [V_particle_1_next, V_particle_2_next]
    iter += 1


  end = time.time()
  return [global_best_solution, global_best_cost, end-start]

#Result Generator
def check_solutions(flow_matrix, cost_matrix, alpha, n_hub):
  cost = []
  comp_time = []
  solutions = []
  for i in range(10):
    result = particle_swarm_heuristic(flow_matrix, cost_matrix, n_hub, alpha)
    best_candidate = result[0]
    best_cost = result[1]
    timer = result[2]
    solutions.append(best_candidate)
    cost.append(best_cost)
    comp_time.append(timer)
    print(best_candidate)
    print(best_cost)
    print(timer)
    print('\n')
  best_cost = min(cost)
  best_solution = solutions[cost.index(min(cost))]
  print('Best Cost:', best_cost)
  print('Best Network:', best_solution)
  print('Optimum hubs:', set(best_solution))
  print('Average Cost:', sum(cost)/10)
  print('Average Time', sum(comp_time)/10)

#Results
check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.2, 3)

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.8, 3)

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.2, 5)

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.8, 5)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.2, 3)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.8, 3)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.2, 5)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.8, 5)

