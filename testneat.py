import os
import pickle
import neat
import gym
import numpy as np
import convlstm
filepath = "/Users/farhanqureshi/Downloads/Amazon.csv"
starting_cash = 3500
att = convlstm.stat(filepath,starting_cash)

# load the winner
with open('winner', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)



observation = [3191.21,3191.72,3192.8,3191.09,1800]

done = False
step = 3711
for step in range(0,len(att.data)):
    action = np.argmax(net.activate(observation))

    observation, reward, done, info = att.sim3(action,step,done)
    print(action)
    
