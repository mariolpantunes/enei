# coding=utf-8

import numpy as np
import random
from numpy import linalg as la


__author__ = "Mário Antunes"
__copyright__ = "Copyright 2016, ENEI Workshop"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "mariolpantunes@gmail.com"


def hill_climb(initial_point, cost_function):
    current_point = initial_point
    step_size = np.ones(initial_point.size)
    acceleration = 1.2
    candidate = np.array([-acceleration, -1.0/acceleration, 0.0, 1/acceleration, acceleration])
    done = False
    costs = [cost_function(initial_point)]
    while not done:
        before = cost_function(current_point)
        for i in xrange(0, current_point.size):
            best = -1
            best_cost = float("inf")
            for j in xrange(0, candidate.size):
                current_point[i] = current_point[i] + step_size[i] * candidate[j]
                temp = cost_function(current_point)
                current_point[i] = current_point[i] - step_size[i] * candidate[j]
                if temp < best_cost:
                    best_cost = temp
                    best = j
            if candidate[best] is 0:
                step_size[i] /= acceleration
            else:
                current_point[i] = current_point[i] + step_size[i] * candidate[best]
                step_size[i] = step_size[i] * candidate[best]
            if (cost_function(current_point) - before) == 0:
                done = True
            costs.append(cost_function(current_point))
    return current_point, costs


def selection(population):
    offspring = []
    size = (len(population)/3)*3
    for r in xrange(0, 3):
        random.shuffle(population)
        for i in xrange(0, size, 3):
            temp = [population[i], population[i+1], population[i+2]]
            temp.sort(key=lambda e: e.fitness(), reverse=True)
            offspring.append(temp[0].crossover(temp[1]))
    return offspring


def avg(l, key):
    a = 0.0
    for e in l:
        a += key(e)
    return a/float(len(l))


def genetic(population, max_it=100, steady_state=.1, prob_mutation=.2):
    size_steady_state = int(len(population) * steady_state)
    number_mutations = int(len(population) * prob_mutation)
    it = 0
    min_costs = [min(population, key=lambda e: e.fitness())]
    avg_costs = [avg(population, key=lambda e: e.fitness())]
    while it < max_it:
        offspring = selection(population)
        offspring.sort(key=lambda e: e.fitness())
        population.sort(key=lambda e: e.fitness(), reverse=True)
        new_generation = []
        new_generation.extend(population[:size_steady_state])
        new_generation.extend(offspring[size_steady_state-1:])
        random.shuffle(new_generation)
        for i in xrange(0, number_mutations):
            new_generation[i].mutation()
        population = new_generation
        it += 1
        min_costs.append(min(population, key=lambda e: e.fitness()))
        avg_costs.append(avg(population, key=lambda e: e.fitness()))
    population.sort(key=lambda e: e.fitness(), reverse=True)
    return population[0], min_costs, avg_costs


def gradient_descent(initial_point, gradient, cost_function, alpha=0.001, max_it=100):
    current_point = initial_point
    done = False
    costs = [cost_function(current_point)]
    it = 0
    while not done:
        initial_point = current_point
        #print "Cost = "+str(cost_function(current_point))
        delta = gradient(initial_point)
        #print "Delta: "+str(delta)
        #print "Alpha x Delta: "+str(np.dot(delta, alpha))
        current_point = np.subtract(initial_point, np.dot(delta, alpha))
        #print "Current Point:"+str(current_point)
        costs.append(cost_function(current_point))
        if it > max_it:
            done = True
        it += 1
    print it
    return current_point, costs
