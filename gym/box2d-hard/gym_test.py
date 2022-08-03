import gym
import pickle
from time import sleep
import numpy as np
import Neural_Net_For_Neat
from gym_trainer import Agent


path = r'C:\Users\Nirkoren\PycharmProjects\NEAT\gym\box2d-hard\lunar_lander_dis\125-fitness=146.243.pickle'
env = gym.make("LunarLander-v2")
observation = env.reset()
f = open(path, 'rb')
agent = pickle.load(f)
f.close()
while True:
  env.render()
  action = agent.take_an_action(observation)
  action = np.argmax(action)
  # action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  if done:
    observation = env.reset()
env.close()

