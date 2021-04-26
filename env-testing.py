import gym


#env = gym.make("BipedalWalker-v2")
env = gym.make("BipedalWalker-v2")

observation = env.reset()
#
# print(observation)
# print(env.action_space)

done = False
while not done:
    observation, reward, done, info = env.step(env.action_space.sample())
    print(observation)
    print(reward)
    print(done)
    print(info)
    #print(env.action_space.sample())

    env.render()
