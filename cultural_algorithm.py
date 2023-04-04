# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 11:38:51 2023

@author: gopal
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from random import randint, uniform
import random

def generate_binary_string(n):
    key1 = ""
 
    # Loop to find the string
    # of desired length
    for i in range(n):
         
        # randint function to generate
        # 0, 1 randomly and converting
        # the result into str
        temp = str(random.randint(0, 1))
 
        # Concatenation the random 0, 1
        # to the final result
        key1 += temp
         
    return(key1)

def func1(num):
    temp = ''
    for i in str(num):
        temp += i
    return temp
        
def b_to_d(num):
    sum = 0 
    n = 8
    for i in num:
        
        sum += int(i) * math.pow(2,n)
        n = n-1
    sum = ((sum/math.pow(2,8)))-1
    if uniform(0, 1) < 0.5:
        return -sum
    return sum

# define the sine function
def sine(x):
    return x*np.sin(10*np.pi*x)+2

# define the fitness function
def fitness_function(population):
    # calculate the fitness of each individual
    fitness = []
    for individual in population:
        i = b_to_d(individual)
        y = sine(i)
        fitness.append(y)
    return fitness

# define the crossover function
def crossover(parent1, parent2):
    # randomly select a crossover point
    p1 = parent1
    p2 = parent2
    crossover_point = randint(0, len(p1) - 1)
    child1 = p1[:crossover_point] +  p2[crossover_point:]
    child2 = p2[:crossover_point] + p1[crossover_point:]
    return child1 , child2

# define the mutation function
def mutation(individual, mutation_rate):
    i = list(individual)
    if uniform(0, 1) < mutation_rate:
        mutate_point = randint(0, len(i) - 1)
        i[mutate_point] = int(not i[mutate_point]) 
    return str(i)

# define the genetic algorithm function
def cultural_algorithm(population_size, num_generations, mutation_rate):
    
    # generate the initial population
    belief_space = []
    population = []
    for i in range(population_size):
        
        b_s = generate_binary_string(8)
        belief_space.append(round(b_to_d(b_s),2))
        if (round(b_to_d(b_s),2) not in belief_space):
            belief_space.append(round(b_to_d(b_s),2))
            population.append(b_s)   
    
    # iterate through the generations
    for generation in range(num_generations):
    
        for i in range (0,len(population)-1,2):
            
            parent1 = population[i]
            parent2 = population[i+1]
            
            # create the offspring using crossover
            child1 ,child2 = crossover(parent1, parent2)

            # apply mutation to the offspring
            #child1 = mutation(child1, mutation_rate)
            #child2 = mutation(child2, mutation_rate)
        
            if (round(b_to_d(child1),2) not in belief_space ):
                population.append(child1)
                belief_space.append(round(b_to_d(child1),2))
            if (round(b_to_d(child2),2) not in belief_space ):
                population.append(child2)
                belief_space.append(round(b_to_d(child2),2))
                
        fitness = fitness_function(population)
        fit = np.argsort(fitness)
        fit1 = fit[::-1]
        newpop=[]
        
        for i in range (len(population)//2):
             newpop.append(population[fit[i]])
             newpop.append(population[fit1[i]])
        newpop = list(set(newpop)) 
        population = newpop[:population_size]
      
    f_pop = [b_to_d(i) for i in population]
    print(len(f_pop))
    # return the best individual in the final population
    return f_pop

# run the genetic algorithm and plot the results
best_individual = cultural_algorithm(population_size=100, num_generations=500 , mutation_rate=0.01)
best_individual.sort()
print(best_individual)
x = np.linspace(-1,1,500)
y = sine(x)
plt.plot(x, y, label="actual")
#plt.scatter(best_individual, sine(np.array(best_individual)))
plt.plot(best_individual, sine(np.array(best_individual)), label="approximation")
plt.legend()
plt.show()
