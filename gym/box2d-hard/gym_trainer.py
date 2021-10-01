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
        self.brain = NeuralNetwork(self.genome, softmax=False, activation=None)
        self.fitness = 0
        self.prev_output = [0 for i in range(4)]

    def take_an_action(self, inputs):
        predictions = self.brain.predict(inputs)
        # if abs(sum(predictions) - sum(self.prev_output)) < 0.000001:
        #     predictions = [np.random.uniform(-1, 1) for i in range(4)]
        # self.prev_output = predictions
        return predictions


class LstmAgent:
    def __init__(self, innov_lst, inputs=0, outputs=0, hidden=0, genome=None):
        if genome is None:
            self.genome1 = Genome(inputs + outputs, outputs, hidden, innov_lst=innov_lst[0], index=0)
            self.genome2 = Genome(inputs + outputs, outputs, hidden, innov_lst=innov_lst[1], index=1)
            self.genome3 = Genome(inputs + outputs, outputs, hidden, innov_lst=innov_lst[2], index=2)
            self.genome4 = Genome(inputs + outputs, outputs, hidden, innov_lst=innov_lst[3], index=3)
        else:
            self.genome1 = Genome(genes=genome[0], index=0)
            self.genome2 = Genome(genes=genome[1], index=1)
            self.genome3 = Genome(genes=genome[2], index=2)
            self.genome4 = Genome(genes=genome[3], index=3)
        self.genome = [self.genome1, self.genome2, self.genome3, self.genome4]
        self.brain = NeuralNetwork(self.genome[0], softmax=False, activation=None)
        self.ignoring = NeuralNetwork(self.genome[1], softmax=False, activation='logistic')
        self.forgetting = NeuralNetwork(self.genome[2], softmax=False, activation='logistic')
        self.selection = NeuralNetwork(self.genome[3], softmax=False, activation='logistic')
        self.fitness = 0
        self.prev_ouput = [0 for i in range(len(self.genome1.genes[0]['output']))]
        self.prev_memory = [0 for i in range(len(self.genome1.genes[0]['output']))]

    def take_an_action(self, inputs):
        combine = list(inputs) + list(self.prev_ouput)
        bt = self.brain.predict(combine)
        it = self.ignoring.predict(combine)
        ft = self.forgetting.predict(combine)
        st = self.selection.predict(combine)
        output = [bt[i] * it[i] for i in range(len(bt))]
        memory = [ft[i] * self.prev_memory[i] for i in range(len(bt))]
        output = [output[i] + memory[i] for i in range(len(bt))]
        self.prev_memory = output
        output = [2 / (1 + np.e ** (-1 * o)) - 1 for o in output]
        output = [output[i] * st[i] for i in range(len(bt))]
        # if abs(sum(self.prev_ouput) - sum(output)) < 0.001:
        #     output = [np.random.uniform(-1, 1) for i in range(4)]
        self.prev_ouput = output
        return output


def evaluate_agent(agent, loop, runs_per_agent, decay, env, q_ag):
    for _ in range(runs_per_agent):
        observation = env.reset()
        while True:
            action = agent.take_an_action(observation)
            observation, reward, done, info = env.step(action)
            if loop < decay:
                agent.fitness += reward + ((0.00035 - (0.00035 * (loop / decay))) * 80 * sum(
                    [np.clip(np.abs(a), 0, 1) for a in action]))
            else:
                agent.fitness += reward
            if done:
                q_ag.put(agent)
                break


def cr_new_file():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + r'\box2d_hard_ag'
    dirs = os.listdir(dir_path)
    if len(dirs) == 0:
        os.mkdir(dir_path + r'\run-1')
        return r'\run-1'
    else:
        os.mkdir(dir_path + rf'\run-{len(dirs) + 1}')
        return rf'\run-{len(dirs) + 1}'


if __name__ == '__main__':
    file_destination = cr_new_file()
    env = gym.make("BipedalWalkerHardcore-v3")
    ne.set_weight_limit(-1, 1)
    ne.training_agent = Agent
    ne.lstm = False
    ne.population_size = 100
    ne.weightCoeff = 2
    ne.DisNExcCoeff = 73
    genome.mutation_rate = 0.1
    genome.crossover_rate = 0.4
    ne.num_of_sensors = 24
    ne.num_of_outputs = 4
    ne.init_gen_0()
    runs_per_agent = 1
    loop = 0
    decay = 200
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
                    observation, reward, done, info = env.step(action)
                    if loop < decay:
                        ag.fitness += reward + ((0.00035 - (0.00035 * (loop / decay))) * 80 * sum([np.clip(np.abs(a), 0, 1) for a in action]))
                    else:
                        ag.fitness += reward
                    if done:
                        break
        print('')
        print("loop -", loop)
        ne.update_fitness_lst()
        print(f'best fitness = {max(ne.fitness_lst)}')
        print(f'avg fitness = {ne.avg_group_fitness(ne.current_gen)}')
        ne.save_agent(r'C:\Users\Nirkoren\PycharmProjects\NEAT\gym\box2d-hard\box2d_hard_ag' + file_destination)
        # best_agent = ne.current_gen[np.argmax(ne.fitness_lst)]
        # observation = env.reset()
        # for _ in range(900):
        #     env.render()
        #     action = best_agent.take_an_action(observation)
        #     observation, reward, done, info = env.step(action)
        #     if done:
        #         break
        ne.create_new_gen()
