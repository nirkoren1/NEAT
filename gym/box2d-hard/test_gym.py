import gym
import pickle
from time import sleep
import numpy as np
import Neural_Net_For_Neat
from gym_trainer import Agent


path = r'D:\Done_Projects\Ai_trader\gym\box2d-hard\box2d_hard\263-fitness=-24.328.pickle'
env = gym.make("BipedalWalkerHardcore-v3")
observation = env.reset()
f = open(path, 'rb')
agent = pickle.load(f)
f.close()
while True:
  env.render()
  action = agent.take_an_action(observation)
  # action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  if done:
    observation = env.reset()
env.close()

