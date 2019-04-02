import sys; sys.path.append('..')

import os
import argparse
from algorithms import TournamentSelection, OnePointCrossover, PointMutation
from algorithms import GEPrune, GEDuplication, GrammaticalEvolution
from grammars import BNFGrammar
from problems import CnnProblem


# disable warning on gpu enable systems
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_arg_parsersed():

    parser = argparse.ArgumentParser(prog='script.py')

    # not optional
    parser.add_argument('grammar', type=str)
    parser.add_argument('dataset', type=str)

    # problem
    parser.add_argument('-sd', '--seed', default=None, type=int)
    parser.add_argument('-ep', '--epochs', default=1, type=int)
    parser.add_argument('-b', '--batch', default=32, type=int)

    # algorithm
    parser.add_argument('-p', '--population', default=5, type=int)
    parser.add_argument('-e', '--evals', default=10, type=int)
    parser.add_argument('-cr', '--crossover', default=0.8, type=float)
    parser.add_argument('-mt', '--mutation', default=0.1, type=float)
    parser.add_argument('-pr', '--prune', default=0.1, type=float)
    parser.add_argument('-dp', '--duplication', default=0.1, type=float)
    parser.add_argument('-mig', '--mingenes', default=2, type=int)
    parser.add_argument('-mag', '--maxgenes', default=10, type=int)
    parser.add_argument('-mp', '--maxprocesses', default=2, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    # parses the arguments
    args = get_arg_parsersed()

    # read grammar and setup parser
    parser = BNFGrammar(args.grammar)

    # problem dataset and parameters
    problem = CnnProblem(parser, args.dataset)
    problem.batch_size = args.batch
    problem.epochs = args.epochs

    # genetic operators to GE
    selection = TournamentSelection(maximize=True)
    crossover = OnePointCrossover(cross_rate=args.crossover)
    mutation = PointMutation(mut_rate=args.mutation,
                             min_value=0, max_value=255)
    prune = GEPrune(prun_rate=args.prune)
    duplication = GEDuplication(dupl_rate=args.duplication)

    # changing ge default parameters
    algorithm = GrammaticalEvolution(problem)
    algorithm.pop_size = args.population
    algorithm.max_evals = args.evals
    algorithm.max_processes = args.maxprocesses
    algorithm.min_genes = args.mingenes
    algorithm.max_genes = args.maxgenes
    algorithm.selection = selection
    algorithm.crossover = crossover
    algorithm.mutation = mutation
    algorithm.prune = prune
    algorithm.duplication = duplication
    algorithm.verbose = args.verbose

    print('--config--')
    print('DATASET', args.dataset)
    print('GRAMMAR', args.grammar)
    print('EPOCHS', args.epochs)
    print('BATCH', args.batch)

    print('POP', args.population)
    print('EVALS', args.evals)

    print('--running--')
    best = algorithm.execute()

    print('--best solution--')
    if best:
        print(best.fitness, best)
    else:
        print('None solution')
