
#Importing Libraries
import numpy as np
import pandas as pd
import random
import itertools
from collections import Counter
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

#Generating the initial solution
def least_cost_initial_solution(cost_matrix, number_hubs):
    number_nodes = cost_matrix.shape[0]
    nodes = range(1, number_nodes + 1)
    random.sample(nodes, number_hubs)
    hubs = random.sample(nodes, number_hubs)
    spokes = [node for node in nodes if node not in hubs]
    initial_solution = [0]*number_nodes
    
    for node in nodes:
        if node in hubs:
            initial_solution[node - 1] = node
        else:
            hub_spoke_cost = {hub : cost_matrix[node - 1][hub - 1] for hub in hubs}
            initial_solution[node - 1] = min(hub_spoke_cost, key = hub_spoke_cost.get)
    
    return initial_solution

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

#Neighbourhoods
def type_1(initial_solution):
    n = len(initial_solution)
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(n)]
    hubs = list(set(solution))
    spokes = [node for node in nodes if node not in hubs]
    num_spokes = dict(Counter(solution))
    hub_single_spoke = list(num_spokes.keys())[list(num_spokes.values()).index(1)]
    random_spoke = random.choice(spokes)
    solution[hub_single_spoke - 1] = random_spoke
    solution[random_spoke - 1] = random_spoke
    return solution


def type_2(initial_solution):
    n = len(initial_solution)
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(n)]
    hubs = list(set(solution))
    spokes = [node for node in nodes if node not in hubs]
    random_hub = random.choice(hubs)
    random_spoke = random.choice(spokes)
    solution = [random_spoke if i == random_hub else i for i in solution]
    solution[random_spoke - 1] = random_spoke
    return solution

def type_3(initial_solution, cost):
    n = len(initial_solution)
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(n)]
    hubs = list(set(solution))
    p = len(hubs)
    spokes = [node for node in nodes if node not in hubs]  
    random_spoke = random.choice(spokes)
    choice_hubs = list(set(hubs) - {solution[random_spoke - 1]})
    hub_spoke_cost = {hub : cost[random_spoke - 1][hub - 1] for hub in choice_hubs}
    hub_spoke_cost = dict(sorted(hub_spoke_cost.items(), key=lambda item: item[1]))
    if p <=4:
      top_hubs = list(hub_spoke_cost.keys())[:(p-1)]
    else: 
      top_hubs = list(hub_spoke_cost.keys())[:4]
    random_hub = random.choice(top_hubs)
    solution[random_spoke - 1] = random_hub

    return solution
    
def type_4(initial_solution, cost):
    n = len(initial_solution)
    solution = initial_solution.copy()
    nodes = [i + 1 for i in range(n)]
    hubs = list(set(solution))
    spokes = [node for node in nodes if node not in hubs]   
    random_hub = random.choice(hubs)
    spokes_of_hub = [i for i in spokes if solution[i-1] == random_hub]
    random_spoke = random.choice(spokes_of_hub)
    solution = [random_spoke if i == random_hub else i for i in solution]
    return solution

#Local Search
def local_search_criteria(array, spoke):
  n_array = array.copy()
  hub = n_array[spoke - 1]
  for i in range(len(n_array)):
    if n_array[i] == hub:
      n_array[i] = spoke
  return n_array

def local_search(array, w, c, alpha):
  best_neighbour = array.copy()
  best_neighbour_cost = network_cost(array, w, c, alpha)
  spokes = [i for i in range(1, len(array)+1) if i not in array]
  for s in spokes:
    neighbour = local_search_criteria(array, s)
    neighbour_cost = network_cost(neighbour, w, c, alpha)
    if neighbour_cost < best_neighbour_cost:
      best_neighbour = neighbour.copy()
      best_neighbour_cost = neighbour_cost
  return best_neighbour, best_neighbour_cost

type_neighbour = [type_4, type_3]

#Main Alogrithm
def simulated_annealing(flow_matrix, cost_matrix, alpha, p, iter):
  start = time.time()
  n = flow_matrix.shape[0]
  initial_solution = least_cost_initial_solution(cost_matrix, p)
  initial_cost = network_cost(initial_solution, flow_matrix, cost_matrix, alpha)
  intitial_threshold = 0.01*initial_cost
  current_solution = initial_solution.copy()
  current_cost = initial_cost
  best_solution = initial_solution.copy()
  best_cost = initial_cost
  K = iter
  gama = 0.9
  threshold = 0.01*initial_cost
  M = (n*p)/10
  threshold = intitial_threshold
  j = 1
  while j<K:
    i = 1
    while i<=M:
      if (1 in dict(Counter(current_solution)).values()):
        neighbour = type_1(current_solution)
        neighbour_cost = network_cost(neighbour, flow_matrix, cost_matrix, alpha)
      elif i == M:
        neighbour = type_2(current_solution)
        neighbour_cost = network_cost(neighbour, flow_matrix, cost_matrix, alpha)
      else:
        neighbour = random.choice(type_neighbour)(current_solution, cost_matrix)
        neighbour_cost = network_cost(neighbour, flow_matrix, cost_matrix, alpha)
      diff = neighbour_cost - current_cost
      if diff <= threshold:
        current_solution = neighbour.copy()
        current_cost = neighbour_cost
        if current_cost < best_cost:
          best_solution = current_solution.copy()
          best_cost = current_cost
      i+=1
    threshold = gama*threshold
    j+=1
  best_solution, best_cost = local_search(best_solution,flow_matrix, cost_matrix, alpha)
  end = time.time()
  return [best_solution, best_cost, end-start]

#Result Generator
def check_solutions(flow_matrix, cost_matrix, alpha, p, iter):
  cost = []
  comp_time = []
  solutions = []
  for i in range(10):
    result = simulated_annealing(flow_matrix, cost_matrix, alpha, p, iter)
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

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.2, 3, 120)

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.8, 3, 120)

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.2, 5, 120)

check_solutions(CAB_10_nodes_flow, CAB_10_nodes_cost, 0.8, 5, 120)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.2, 3, 120)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.8, 3, 120)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.2, 5, 120)

check_solutions(CAB_25_nodes_flow, CAB_25_nodes_cost, 0.8, 5, 120)

check_solutions(TR_55_nodes_flow, TR_55_nodes_cost, 0.2, 3, 120)

check_solutions(TR_55_nodes_flow, TR_55_nodes_cost, 0.8, 3, 120)

check_solutions(TR_55_nodes_flow, TR_55_nodes_cost, 0.2, 5, 120)

check_solutions(TR_55_nodes_flow, TR_55_nodes_cost, 0.8, 5, 120)

check_solutions(TR_81_nodes_flow, TR_81_nodes_cost, 0.2, 5, 120)

check_solutions(TR_81_nodes_flow, TR_81_nodes_cost, 0.8, 5, 120)

check_solutions(TR_81_nodes_flow, TR_81_nodes_cost, 0.2, 7, 120)

check_solutions(TR_81_nodes_flow, TR_81_nodes_cost, 0.8, 7, 120)

check_solutions(RGP_100_nodes_flow, RGP_100_nodes_cost, 0.2, 7, 120)

check_solutions(RGP_100_nodes_flow, RGP_100_nodes_cost, 0.8, 7, 120)

check_solutions(RGP_100_nodes_flow, RGP_100_nodes_cost, 0.2, 10, 120)

check_solutions(RGP_100_nodes_flow, RGP_100_nodes_cost, 0.8, 10, 120)

check_solutions(RGP_130_nodes_flow, RGP_130_nodes_cost, 0.2, 7, 120)

check_solutions(RGP_130_nodes_flow, RGP_130_nodes_cost, 0.8, 7, 120)

check_solutions(RGP_130_nodes_flow, RGP_130_nodes_cost, 0.2, 10, 120)

check_solutions(RGP_130_nodes_flow, RGP_130_nodes_cost, 0.8, 10, 120)

