import Optimization as opt
import matplotlib.pyplot as plt
import numpy as np
import random


def cost_function(point):
    return (np.power(point[0] - 3., 2.) - 5.)+5


def gradient(point):
    return 2. * point[0] - 6.


class OptimizationChromosome:
    def __init__(self):
        self.point = np.array([random.uniform(-10.0, 10.0)])

    def fitness(self):
        return 1./(cost_function(self.point)+1.)

    def crossover(self, chromosome):
        c = OptimizationChromosome()
        c.point[0] = (self.point[0] + chromosome.point[0])/2.
        return c

    def mutation(self):
        self.point = np.array([self.point[0]+random.uniform(0.0, 1.0)])

    def __str__(self):
        return "Point -> "+str(self.point)+" fitness -> "+str(self.fitness())

    def __repr__(self):
        return self.__str__()

    def __float__(self):
        return float(self.fitness())


initial_point = np.array([10.])
print "Hill climb optimization"
min_point, costs = opt.hill_climb(initial_point, cost_function)
print "Minimum point ->"+str(min_point)
print "Cost function ->"+str(cost_function(min_point))
print ""
plt.plot(costs, color='blue')
plt.show()

population = [OptimizationChromosome() for i in xrange(0, 10)]
print "Genetic optimization"
best_chromosome, min_cost, avg_cost = opt.genetic(population)
print "Minimum point ->"+str(best_chromosome)
print "Cost function ->"+str(cost_function(best_chromosome.point))
print ""
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
ax1.plot(min_cost, color='red')
ax1.set_title('Fitness')
ax2.plot(avg_cost, color='blue')
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.show()

print "Gradient descent optimization"
min_point, costs = opt.gradient_descent(initial_point, gradient, cost_function)
print "Minimum point ->"+str(min_point)
print "Cost function ->"+str(cost_function(min_point))
print ""
plt.plot(costs, color='blue')
plt.show()
