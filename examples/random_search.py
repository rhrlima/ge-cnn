import sys; sys.path.append('..')

import os
import argparse
from algorithms import RandomSearch
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
    parser.add_argument('-mp', '--maxprocesses', default=2, type=int)
    parser.add_argument('-e', '--evals', default=10, type=int)
    parser.add_argument('-mig', '--mingenes', default=2, type=int)
    parser.add_argument('-mag', '--maxgenes', default=10, type=int)

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

    # changing pge default parameters
    algorithm = RandomSearch(problem)
    algorithm.min_value = 0
    algorithm.max_value = 255
    algorithm.min_size = args.mingenes
    algorithm.max_size = args.maxgenes
    algorithm.max_evals = args.evals
    algorithm.max_processes = args.maxprocesses

    print('--config--')
    print('DATASET', args.dataset)
    print('GRAMMAR', args.grammar)

    print('--running--')
    best = algorithm.execute()

    print('--best solution--')
    if best:
        print(best.fitness, best)
    else:
        print('None solution')
