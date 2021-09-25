from random import sample
from Connection_gene import ConGene
import numpy as np
from functools import reduce

mutation_rate = 0.05
crossover_rate = 0.4
low_weight = 0
high_weight = 1


class Genome:
    def __init__(self, inputs=0, outputs=0, hidden=0, random=True, genes=None, innov_lst=None, index=0):
        if genes is None:
            self.inputs = inputs
            self.outputs = outputs
            self.hidden = hidden
            self.node_genes = {"sensor": [], 'output': [], "hidden": []}
            counter = 0
            for i in range(self.inputs):
                counter += 1
                self.node_genes["sensor"].append(counter)
            for o in range(self.outputs):
                counter += 1
                self.node_genes['output'].append(counter)
            self.connect_genes = []
            for s in self.node_genes['sensor']:
                for o in self.node_genes['output']:
                    self.connect_genes.append(ConGene(in_node=s, out_node=o, innov_lst=innov_lst, index=index))
        elif len(genes) != 2:
            print("genes must be a tuple of 2 elements")
            raise ValueError
        else:
            self.node_genes = genes[0]
            self.connect_genes = genes[1]
        self.genes = [self.node_genes, self.connect_genes]


def connection_enable_mutate(genome):
    genes = genome.copy()
    idx = np.random.randint(low=0, high=len(genes[1]))
    enabled = genes[1][idx].enabled
    genes[1][idx].enabled = not enabled
    return genes


def add_connection_mutate(genome, innov_lst):
    genes = genome.copy()
    counter = 0
    if len(genes[0]["hidden"]) == 0:
        return genes
    else:
        limit = len(genes[0]["sensor"]) * len(genes[0]["output"]) * len(genes[0]["hidden"])
    while counter < 2 * limit:
        counter += 1
        list_of_random_items = sample(genes[0]["sensor"] + genes[0]["output"] + genes[0]["sensor"], 2)
        first_random_node = list_of_random_items[0]
        second_random_node = list_of_random_items[1]
        connected = False
        for g in genes[1]:
            if (g.in_node == first_random_node and g.out_node == second_random_node) or (
                    g.in_node == second_random_node and g.out_node == first_random_node):
                connected = True
                break
        if connected:
            break
        else:
            l1 = [first_random_node]
            first_to_second = False
            c = 0
            while not first_to_second and len(l1) != 0:
                c += 1
                if c > 100:
                    # print(genes[0], [(a.in_node, a.out_node, a.enabled, a.weight, a.innov) for a in genes[1]])
                    # print(first_random_node)
                    # print(second_random_node)
                    # exit(0)
                    genes[1].append(ConGene(in_node=second_random_node, out_node=first_random_node, innov_lst=innov_lst))
                    return genes
                l1 = [gen.out_node for gen in genes[1] if gen.in_node in l1]
                if second_random_node in l1:
                    genes[1].append(ConGene(in_node=first_random_node, out_node=second_random_node, innov_lst=innov_lst))
                    return genes
    return genes


def weight_mutate(genome):
    genes = genome.copy()
    genes[1][np.random.randint(low=0, high=len(genes[1]))].weight = np.random.uniform(low_weight, high_weight)
    return genes


def node_mutate(genome, innov_lst):
    genes = genome.copy()
    combined_node_genes = genes[0]["sensor"] + genes[0]["output"] + genes[0]["hidden"]
    idx = np.random.randint(low=0, high=len(genes[1]))
    genes[1][idx].enabled = False
    in_node = genes[1][idx].in_node
    out_node = genes[1][idx].out_node
    new_node_id = max(combined_node_genes) + 1
    genes[0]["hidden"].append(new_node_id)
    genes[1].append(ConGene(in_node=in_node, out_node=new_node_id, innov_lst=innov_lst))
    genes[1].append(ConGene(in_node=new_node_id, out_node=out_node, innov_lst=innov_lst))
    return genes


def mutate(genome, innov_lst):
    rand = np.random.random()
    if rand < mutation_rate:
        rand = np.random.random()
        if rand < 0.8:
            genes = weight_mutate(genome)
        elif 0.8 < rand < 0.85:
            genes = add_connection_mutate(genome, innov_lst)
        elif 0.85 < rand < 0.9:
            genes = node_mutate(genome, innov_lst)
        elif rand > 0.95:
            genes = connection_enable_mutate(genome)
        else:
            return genome
        return genes
    else:
        return genome


def mutate_lstm(genome, innov_lst):
    # genome = [a.genes for a in genome]
    rand = np.random.random()
    if rand < mutation_rate:
        rand_idx = int(np.floor(np.random.random() * 4))
        rand = np.random.random()
        if rand < 0.8:
            genome[rand_idx] = weight_mutate(genome[rand_idx])
        elif 0.8 < rand < 0.85:
            genome[rand_idx] = add_connection_mutate(genome[rand_idx], innov_lst[rand_idx])
        elif 0.85 < rand < 0.87:
            genome[rand_idx] = node_mutate(genome[rand_idx], innov_lst[rand_idx])
        elif rand > 0.95:
            genome[rand_idx] = connection_enable_mutate(genome[rand_idx])
        else:
            return genome
        return genome
    else:
        return genome


def lstm_crossover(parent1, parent2):
    new_genes = [[] for i in range(4)]
    for i in range(4):
        new_genes[i] = [None, []]
        if parent1.fitness > parent2.fitness:
            fitter_parent = parent1
            les_fitter_parent = parent2
        else:
            fitter_parent = parent2
            les_fitter_parent = parent1
        new_genes[i][0] = fitter_parent.genome[i].genes[0].copy()
        worse_genes = [h.innov for h in les_fitter_parent.genome[i].genes[1]]
        for gen in fitter_parent.genome[i].genes[1]:
            if gen.innov in worse_genes:
                if np.random.random() < 0.5:
                    new_genes[i][1].append(gen)
                else:
                    new_genes[i][1].append(les_fitter_parent.genome[i].genes[1][worse_genes.index(gen.innov)])
                if not gen.enabled or not les_fitter_parent.genome[i].genes[1][worse_genes.index(gen.innov)].enabled:
                    if np.random.random() < 0.75:
                        new_genes[i][1][-1].enabled = False
            else:
                new_genes[i][1].append(gen)
    return new_genes


def crossover(parent1, parent2):
    if parent1.fitness > parent2.fitness:
        fitter_parent = parent1
        les_fitter_parent = parent2
    else:
        fitter_parent = parent2
        les_fitter_parent = parent1
    rand = np.random.random()
    if rand < crossover_rate:
        new_genes = [None, []]
        new_genes[0] = fitter_parent.genome.genes[0].copy()
        worse_genes = [h.innov for h in les_fitter_parent.genome.genes[1]]
        for gen in fitter_parent.genome.genes[1]:
            if gen.innov in worse_genes:
                if np.random.random() < 0.5:
                    new_genes[1].append(gen)
                else:
                    new_genes[1].append(les_fitter_parent.genome.genes[1][worse_genes.index(gen.innov)])
                if not gen.enabled or not les_fitter_parent.genome.genes[1][worse_genes.index(gen.innov)].enabled:
                    if np.random.random() < 0.75:
                        new_genes[1][-1].enabled = False
            else:
                new_genes[1].append(gen)
        return new_genes
    else:
        return fitter_parent.genome.genes


def calc_weight_diff(genome1, genome2):
    diff = 0
    matching = 0
    innov_p2 = [h.innov for h in genome2.genes[1]]
    for gen in genome1.genes[1]:
        if gen.innov in innov_p2:
            matching += 1
            diff += np.abs(gen.weight - genome2.genes[1][innov_p2.index(gen.innov)].weight)
    if matching == 0:
        return 100
    else:
        return diff / matching


def disjoint_and_excess_diff(genome1, genome2):
    matching = 0
    innov_p1 = [h.innov for h in genome2.genes[1]]
    for gen in genome1.genes[1]:
        if gen.innov in innov_p1:
            matching += 1
    return len(genome1.genes[1]) + len(genome2.genes[1]) - (2 * matching)
