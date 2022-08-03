import Connection_gene
import genome
import numpy as np
from genome import disjoint_and_excess_diff, calc_weight_diff, mutate, crossover, mutate_lstm, lstm_crossover
from random import sample
import pickle
import os


num_of_sensors = 5
num_of_outputs = 3
tournament_group = 4
population_size = 100
elitism_group = 10
DisNExcCoeff = 1  # 1
weightCoeff = 2  # 2
diffThreh = 1.5  # 1.5
lstm = False
current_gen = []
next_gen = []
fitness_lst = []
black_lst_gen = []
if not lstm:
    innov_lst = []
else:
    innov_lst = [[] for i in range(4)]
species = []
training_agent = None


def set_weight_limit(low, high):
    genome.low_weight = low
    genome.high_weight = high
    Connection_gene.low_weight = low
    Connection_gene.high_weight = high


def avg_group_fitness(group):
    total = 0
    n = 0
    for offspring in group:
        total += offspring.fitness
        n += 1
    return total / n


def calc_species_count():
    global next_gen
    if len(species) == 0:
        print("There's not species yet")
        raise ValueError
    else:
        for spec in species:
            ln = len(spec)
            for ag in spec:
                ag.fitness /= ln
        counts = []
        pop_avg_fitness = avg_group_fitness(current_gen)
        left = population_size - len(next_gen)
        for spec in species:
            count = np.ceil((avg_group_fitness(spec) / pop_avg_fitness) * len(spec))
            if is_banned(spec):
                count = 0
            left -= count
            if left < 0:
                count += left
                left = 0
            counts.append(count)
        return counts


def generation_to_species():
    if not lstm:
        for agent in current_gen:
            species_found = False
            if len(species) == 0:
                species.append([agent])
            else:
                for spec in species:
                    rep = spec[int(np.floor(np.random.random() * len(spec)))]
                    n = max(len(rep.genome.genes[1]), len(agent.genome.genes[1])) - 20
                    if n < 1:
                        n = 1
                    else:
                        n += 20
                    diff = (DisNExcCoeff * disjoint_and_excess_diff(agent.genome, rep.genome) / n) + weightCoeff * calc_weight_diff(agent.genome, rep.genome)
                    if diff < diffThreh:
                        species_found = True
                        spec.append(agent)
                        break
                if not species_found:
                    species.append([agent])
    else:
        for agent in current_gen:
            species_found = False
            if len(species) == 0:
                species.append([agent])
            else:
                for spec in species:
                    rep = spec[int(np.floor(np.random.random() * len(spec)))]
                    diff = 0
                    for i in range(4):
                        n = max(len(rep.genome[i].genes[1]), len(agent.genome[i].genes[1])) - 20
                        if n < 1:
                            n = 1
                        else:
                            n += 20
                        diff += (DisNExcCoeff * disjoint_and_excess_diff(agent.genome[i], rep.genome[i]) / n) + weightCoeff * calc_weight_diff(agent.genome[i], rep.genome[i])
                    if diff < diffThreh * 4:
                        species_found = True
                        spec.append(agent)
                        break
                if not species_found:
                    species.append([agent])


def update_fitness_lst():
    if len(fitness_lst) == 0:
        for ob in current_gen:
            fitness_lst.append(ob.fitness)
    else:
        for ob_idx in range(len(current_gen)):
            fitness_lst[ob_idx] = current_gen[ob_idx].fitness


def is_banned(spec):
    best_spec_ag = spec[np.argmax([ag.fitness for ag in spec])]
    if len(black_lst_gen) == 0:
        black_lst_gen.append({'agent': best_spec_ag, 'age': 1, 'fitness': best_spec_ag.fitness, 'dead': False})
        return False
    else:
        species_found = False
        for sp in black_lst_gen:
            rep = sp['agent']
            n = max(len(rep.genome.genes[1]), len(best_spec_ag.genome.genes[1])) - 20
            if n < 1:
                n = 1
            else:
                n += 20
            diff = (DisNExcCoeff * disjoint_and_excess_diff(best_spec_ag.genome, rep.genome) / n) + weightCoeff * calc_weight_diff(best_spec_ag.genome, rep.genome)
            if diff < diffThreh:
                species_found = True
                if sp['dead']:
                    return True
                elif best_spec_ag.fitness > sp['fitness']:
                    sp['agent'] = best_spec_ag
                    sp['age'] = 1
                    sp['fitness'] = best_spec_ag.fitness
                    return False
                else:
                    sp['age'] += 1
                    if sp['age'] > 14:
                        sp['dead'] = True
                        return False
        if not species_found:
            black_lst_gen.append({'agent': best_spec_ag, 'age': 1, 'fitness': best_spec_ag.fitness, 'dead': False})
            return False


def elitism():
    global species
    if not lstm:
        for spec in species:
            if len(spec) > 4:
                best_ag = spec[np.argmax([ag.fitness for ag in spec])]
                best_ag.fitness = 0
                next_gen.append(best_ag)
                next_gen.append(training_agent(innov_lst=innov_lst, genome=mutate(best_ag.genome.genes, innov_lst)))
    else:
        for spec in species:
            if len(spec) > 4:
                best_ag = spec[np.argmax([ag.fitness for ag in spec])]
                next_gen.append(best_ag)
                next_gen.append(training_agent(innov_lst=innov_lst, genome=mutate_lstm([best_ag.genome[a].genes for a in range(4)], innov_lst)))


def init_gen_0():
    global innov_lst
    for i in range(population_size):
        current_gen.append(training_agent(innov_lst=innov_lst, inputs=num_of_sensors, outputs=num_of_outputs))


def create_new_gen():
    global innov_lst, next_gen, current_gen, fitness_lst, species
    next_gen = []
    generation_to_species()
    elitism()
    print(f'species num =  {len(species)}')
    # if abs(avg_group_fitness(current_gen) - 0) > 0.00001:
    species_counts = calc_species_count()
    # else:
    #     print("Low fitness - initializing...")
    #     species_counts = [np.floor(100 / len(species)) for i in range(len(species))]
    #     species_counts[-1] -= elitism_group * 2
    print(f"species count = {sum(species_counts)}")
    for spec_indx in range(len(species)):
        for count in range(int(species_counts[spec_indx])):
            parent1, parent2 = tournament_selection(species[spec_indx])
            if not lstm:
                next_gen.append(training_agent(innov_lst=innov_lst, genome=mutate(crossover(parent1, parent2), innov_lst)))
            else:
                next_gen.append(training_agent(innov_lst=innov_lst, genome=mutate_lstm(lstm_crossover(parent1, parent2), innov_lst)))
    for ag in next_gen:
        ag.fitness = 0
    for i in range(len(fitness_lst)):
        fitness_lst[i] = 0
    print(f"agents left = {int(population_size - len(next_gen))}")
    next_gen = next_gen[:population_size]
    for g in range(int(population_size - len(next_gen))):
        next_gen.append(training_agent(innov_lst=innov_lst, inputs=num_of_sensors, outputs=num_of_outputs))
    current_gen = next_gen.copy()
    species = []
    next_gen = []
    print(f"fitness len - {len(fitness_lst)}  current gen len - {len(current_gen)}\n")


def tournament_selection(spec):
    if tournament_group > len(spec):
        samples = sample(range(len(spec)), len(spec))
    else:
        samples = sample(range(len(spec)), tournament_group)
    fitnesses = [spec[s].fitness for s in samples]
    args = list(reversed(np.argsort(fitnesses)))
    best = spec[args[0]]
    if len(args) == 1:
        second = best
    else:
        second = spec[args[1]]
    return best, second


def step_generation(data_input):
    for ag in current_gen:
        ag.take_an_action(data_input)


def save_agent(path):
    best_agent = current_gen[np.argmax(fitness_lst)]
    files = os.listdir(path)
    with open(path + f"/{len(files) + 1}-fitness=" + str(max(fitness_lst))[:7] + '.pickle', 'wb') as outp:
        pickle.dump(best_agent, outp, pickle.HIGHEST_PROTOCOL)
    print(f"Agent saved with {max(fitness_lst)} fitness")


if __name__ == '__main__':
    pass
