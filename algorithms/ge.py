import time
from multiprocessing import Pool
import numpy as np
from .solutions import GESolution
from .ea import BaseEvolutionaryAlgorithm


class GrammaticalEvolution(BaseEvolutionaryAlgorithm):

    def __init__(self, problem):
        super(GrammaticalEvolution, self).__init__(problem)

        self.seed = None

        self.pop_size = 5
        self.max_evals = 10

        self.min_genes = 1
        self.max_genes = 10
        self.min_value = 0
        self.max_value = 255

        self.selection = None
        self.crossover = None
        self.mutation = None
        self.prune = None
        self.duplication = None

        self.population = None
        self.evals = None

    def create_solution(self, min_size, max_size, min_value, max_value):

        if min_size >= max_size:
            raise ValueError('[create solution] min >= max')

        genes = np.random.randint(min_value, max_value, np.random.randint(
            min_size, max_size))

        return GESolution(genes)

    def create_population(self, size):
        population = []
        for i in range(size):
            solution = self.create_solution(self.min_genes, self.max_genes,
                                            self.min_value, self.max_value)
            population.append(solution)
        return population

    def evaluate_solution(self, solution):

        if not solution.evaluated:
            fitness, model = self.problem.evaluate(solution)
        else:
            fitness, model = solution.fitness, solution.phenotype

        return fitness, model

    def evaluate_population(self, population):

        pool = Pool(processes=self.max_processes)
        result = pool.map_async(self.evaluate_solution, population)

        pool.close()
        pool.join()

        for sol, res in zip(population, result.get()):
            sol.fitness, sol.phenotype = res
            sol.evaluated = True

        self.population.sort(key=lambda x: x.fitness, reverse=self.maximize)

    def replace(self, population, offspring):

        population += offspring
        for _ in range(len(offspring)):
            population.pop()

    def execute(self):

        self.population = self.create_population(self.pop_size)
        self.evaluate_population(self.population)
        self.evals = len(self.population)

        self.print_progress()

        while self.evals < self.max_evals:

            offspring_pop = []

            for index in range(self.pop_size):
                parents = self.selection.execute(self.population)
                offspring = self.crossover.execute(parents)
                self.mutation.execute(offspring)
                self.prune.execute(offspring)
                self.duplication.execute(offspring)
                offspring_pop += offspring

            self.evaluate_population(offspring_pop)
            self.replace(self.population, offspring_pop)
            self.evals += len(offspring_pop)

            self.print_progress()

        return self.population[0]

    def print_progress(self):
        curr_time = time.strftime('%x %X')
        best = self.population[0].genotype
        best_fit = self.population[0].fitness

        print(f'<{curr_time}> evals: {self.evals}/{self.max_evals} \t\
            best so far: {best}\tfitness: {best_fit}')
