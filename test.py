import os
import pickle
import neat
import gym
import numpy as np
import convlstm
import matplotlib.pyplot as plt
filepath = "/Users/farhanqureshi/Downloads/Bitcoin.csv"
starting_cash = 1000
att = convlstm.stat(filepath,starting_cash)
gprof = []
act_price = []
unit = []
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
#step = 3711
i =0
for step in range(3711,len(att.data)):
    action = np.argmax(net.activate(observation))

    observation, reward, done, info = att.sim3(action,step,done)
    print(action)
    gprof.append(att.profit_cal())
    act_price.append(att.data["Open"][step])
    unit.append(i)
    i = i+1
print(att.profit_cal())
plt.plot( unit, gprof)
plt.plot( unit, act_price)
plt.legend()
plt.show()
