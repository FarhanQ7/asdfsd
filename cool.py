# pip3 install gym
# pip3 install neat-python

# for gym stuff:
# apt install xvfb ffmpeg xorg-dev libsdl2-dev swig cmake
# pip3 install gym[box2d]

import multiprocessing
import os
import pickle

import neat
import numpy as np
#import cart_pole
# import gym
import convlstm



runs_per_net = 2
filepath = "/Users/farhanqureshi/Downloads/Amazon.csv"
starting_cash = 3500
#att = convlstm.stat(filepath,starting_cash)
# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    att = convlstm.stat(filepath,starting_cash)
    fitnesses = []
    step = 0
    for i in range(0,runs_per_net):


        observation = [3031.74,3028.58,3037.8,3023,501247]
        fitness = 0.0
        done = False
        for step in range(0,int(len(att.data)*.5)):

            action = np.argmax(net.activate(observation))
            observation, reward, done, info = att.sim2(action,step,done)
            fitness += reward*.01



        fitnesses.append(fitness)
    print(np.mean(fitnesses))
    return np.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    winner = pop.run(pe.evaluate,300)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)




if __name__ == '__main__':
    run()
