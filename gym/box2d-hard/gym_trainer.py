import Neural_Net_For_Neat
from Neural_Net_For_Neat import NeuralNetwork
from genome import *
import genome
import numpy as np
import neat as ne
import gym
import sys
import pickle
import winsound
from multiprocessing import Queue, Pool, Manager, Process
import os
from time import sleep


class Agent:
    def __init__(self, innov_lst, inputs=0, outputs=0, hidden=0, genome=None):
        if genome is None:
            self.genome = Genome(inputs, outputs, hidden, innov_lst=innov_lst)
        else:
            self.genome = Genome(genes=genome)
        self.brain = NeuralNetwork(self.genome, softmax=True, activation=None)
        self.fitness = 0
        self.prev_output = [0 for i in range(4)]

    def take_an_action(self, inputs):
        predictions = self.brain.predict(inputs)
        # if abs(sum(predictions) - sum(self.prev_output)) < 0.000001:
        #     predictions = [np.random.uniform(-1, 1) for i in range(4)]
        # self.prev_output = predictions
        return predictions


def evaluate_agent(agent, loop, runs_per_agent, decay, env, q_ag):
    for _ in range(runs_per_agent):
        observation = env.reset()
        while True:
            action = agent.take_an_action(observation)
            observation, reward, done, info = env.step(action)
            # if loop < decay:
            #     agent.fitness += reward + ((0.00035 - (0.00035 * (loop / decay))) * 80 * sum(
            #         [np.clip(np.abs(a), 0, 1) for a in action]))
            # else:
            agent.fitness += reward
            if done:
                q_ag.put(agent)
                break


def cr_new_file():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + r'\lunar_lander_dis'
    dirs = os.listdir(dir_path)
    if len(dirs) == 0:
        os.mkdir(dir_path + r'\run-1')
        return dir_path + r'\run-1'
    else:
        os.mkdir(dir_path + rf'\run-{len(dirs) + 1}')
        return dir_path + rf'\run-{len(dirs) + 1}'


if __name__ == '__main__':
    file_destination = cr_new_file()
    print(file_destination)
    env = gym.make("LunarLander-v2")
    ne.set_weight_limit(0, 1)
    ne.training_agent = Agent
    ne.lstm = False
    ne.population_size = 500
    ne.weightCoeff = 2
    ne.DisNExcCoeff = 73
    genome.mutation_rate = 0.07
    genome.crossover_rate = 0.4
    ne.num_of_sensors = env.observation_space.shape[0]
    ne.num_of_outputs = 4
    ne.init_gen_0()
    runs_per_agent = 50
    loop = 0
    decay = 200
    best_fitness = - np.inf
    # simpler agent for a good start
    # path = r'C:\Users\Nirkoren\PycharmProjects\Ai_trader\gym\box2d-v0\box2d_ag\993-fitness=43.0087.pickle'
    # f = open(path, 'rb')
    # simpler_agent = pickle.load(f)
    # f.close()
    # for ag in range(ne.population_size):
    #     ne.current_gen.append(Agent(innov_lst=ne.innov_lst, genome=mutate(simpler_agent.genome.genes, ne.innov_lst)))
    while True:
        loop += 1
        print("start new gen")
        # m = Manager()
        # q_ag = m.Queue()
        # evaluations = 0
        # workers = [Process(target=evaluate_agent, args=(ag, loop, runs_per_agent, decay, env, q_ag)) for ag in ne.current_gen]
        # for work in workers:
        #     work.start()
        #     evaluations += 1
        #     if evaluations - ne.population_size in [-15, -10, -5]:
        #         winsound.Beep(300, 500)
        #     if evaluations - ne.population_size in [-2]:
        #         winsound.Beep(400, 500)
        #     if evaluations % 5 == 0:
        #         sleep(2)
        #     if evaluations % 50 == 0:
        #         sleep(10)
        #     sys.stdout.write(f"\r{evaluations}/{ne.population_size}")
        #     sys.stdout.flush()
        # for work in workers:
        #     work.join()
        # ne.current_gen = []
        # while True:
        #     try:
        #         ne.current_gen.append(q_ag.get(timeout=1))
        #     except:
        #         break
        for ag in ne.current_gen:
            sys.stdout.write(f"\r{ne.current_gen.index(ag)}/{ne.population_size - 1}")
            sys.stdout.flush()
            for _ in range(runs_per_agent):
                observation = env.reset()
                while True:
                    action = ag.take_an_action(observation)
                    observation, reward, done, info = env.step(np.argmax(action))
                    # if loop < decay:
                    #     ag.fitness += reward + ((0.00035 - (0.00035 * (loop / decay))) * 80 * sum([np.clip(np.abs(a), 0, 1) for a in action]))
                    # else:
                    ag.fitness += reward
                    if done:
                        break
                ag.fitness /= runs_per_agent
        print('')
        # print("loop -", loop)
        ne.update_fitness_lst()
        # print(f'best fitness = {max(ne.fitness_lst)}')
        # print(f'avg fitness = {ne.avg_group_fitness(ne.current_gen)}')
        print(f"loop - {loop}  best fitness = {max(ne.fitness_lst):.2f} avg fitness = "
              f"{ne.avg_group_fitness(ne.current_gen):.2f}")
        if max(ne.fitness_lst) > best_fitness:
            best_fitness = max(ne.fitness_lst)
            ne.save_agent(file_destination)
        # best_agent = ne.current_gen[np.argmax(ne.fitness_lst)]
        # observation = env.reset()
        # for _ in range(900):
        #     env.render()
        #     action = best_agent.take_an_action(observation)
        #     observation, reward, done, info = env.step(action)
        #     if done:
        #         break
        ne.create_new_gen()
