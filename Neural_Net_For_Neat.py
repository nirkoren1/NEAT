from node import Node


class NeuralNetwork:
    def __init__(self, genome=None, softmax=True, activation='sigmoid'):
        self.softmax_bool = softmax
        self.activation = activation
        if self.activation not in ['sigmoid', 'logistic', None]:
            print("Not valid activation function")
            raise ValueError
        self.problematic = False
        if genome is None:
            print("You must enter a genome")
            raise ValueError
        # reads the genome and creates NN
        else:
            if type(genome) == list:
                self.genome = genome
            else:
                self.genome = genome.genes
            self.nodes_lst = self.genome[0]["sensor"] + self.genome[0]["output"] + self.genome[0]["hidden"]
            self.sensor_nodes = [Node(n_id) for n_id in self.genome[0]["sensor"]]
            self.hidden_nodes = [Node(n_id) for n_id in self.genome[0]["hidden"]]
            self.output_nodes = [Node(n_id) for n_id in self.genome[0]["output"]]
            self.all_nodes = self.sensor_nodes + self.output_nodes + self.hidden_nodes
            self.h_plus_o = self.output_nodes + self.hidden_nodes
            self.sum_all_connections = 0
            for node in self.h_plus_o:
                for g in self.genome[1]:
                    if (node.id == g.out_node) and (g.enabled == True):
                        self.sum_all_connections += 1
                        node.n_of_connections += 1
            for node in self.hidden_nodes:
                subtract = True
                for g in self.genome[1]:
                    if node.id == g.out_node and g.enabled is True:
                        subtract = False
                if subtract:
                    l1 = [g for g in self.genome[1] if (node.id == g.in_node and g.enabled is True)]
                    self.sum_all_connections -= len(l1)
                    for gen in l1:
                        self.all_nodes[self.nodes_lst.index(gen.out_node)].n_of_connections -= 1
            self.sum_all_connections = sum([i.n_of_connections for i in self.all_nodes])

    @staticmethod
    def softmax(outputs: list):
        if sum(outputs) == 0:
            return [out for out in outputs]
        else:
            return [out / sum(outputs) for out in outputs]

    def fix_net(self):
        self.nodes_lst = self.genome[0]["sensor"] + self.genome[0]["output"] + self.genome[0]["hidden"]
        self.sensor_nodes = [Node(n_id) for n_id in self.genome[0]["sensor"]]
        self.hidden_nodes = [Node(n_id) for n_id in self.genome[0]["hidden"]]
        self.output_nodes = [Node(n_id) for n_id in self.genome[0]["output"]]
        self.all_nodes = self.sensor_nodes + self.output_nodes + self.hidden_nodes
        self.h_plus_o = self.output_nodes + self.hidden_nodes
        self.sum_all_connections = 0
        for node in self.h_plus_o:
            for g in self.genome[1]:
                if (node.id == g.out_node) and (g.enabled == True):
                    self.sum_all_connections += 1
                    node.n_of_connections += 1
        for node in self.hidden_nodes:
            subtract = True
            for g in self.genome[1]:
                if node.id == g.out_node and g.enabled is True:
                    subtract = False
            if subtract:
                l1 = [g for g in self.genome[1] if (node.id == g.in_node and g.enabled is True)]
                self.sum_all_connections -= len(l1)
                for gen in l1:
                    self.all_nodes[self.nodes_lst.index(gen.out_node)].n_of_connections -= 1
        self.sum_all_connections = sum([i.n_of_connections for i in self.all_nodes])

    def predict(self, inputs):
        self.nodes_lst = self.genome[0]["sensor"] + self.genome[0]["output"] + self.genome[0]["hidden"]
        # set sensor nodes from inputs
        if len(inputs) != len(self.sensor_nodes):
            print(f"input len {len(inputs)} don't match the number of sensor nodes {len(self.sensor_nodes)}")
            raise ValueError
        else:
            for s in range(len(self.sensor_nodes)):
                self.sensor_nodes[s].set_value(inputs[s])
            for g in self.genome[1]:
                if g.in_node in self.genome[0]["sensor"] and g.out_node not in self.genome[0]["sensor"] and g.enabled is True:
                    try:
                        out_node = self.all_nodes[self.nodes_lst.index(g.out_node)]
                    except:
                        self.fix_net()
                        out_node = self.all_nodes[self.nodes_lst.index(g.out_node)]
                    in_node = self.all_nodes[self.nodes_lst.index(g.in_node)]
                    out_node.input_values.append(in_node.itself_value * g.weight)
                # elif g.in_node in self.genome[0]["sensor"] and g.out_node in self.genome[0]["output"] and g.enabled is True:
                #     out_node = self.all_nodes[self.nodes_lst.index(g.out_node)]
                #     in_node = self.all_nodes[self.nodes_lst.index(g.in_node)]
                #     out_node.input_values.append(in_node.itself_value * g.weight)
            all_inputs = sum([len(n.input_values) for n in self.h_plus_o])
            c = 0
            while self.sum_all_connections > all_inputs:
                c += 1
                if c > 100:
                    # print(self.genome[0], [(a.in_node, a.out_node, a.enabled, a.weight, a.innov) for a in self.genome[1]])
                    # print(self.sum_all_connections, all_inputs)
                    self.problematic = True
                    for o in self.output_nodes:
                        o.calculate_val()
                        if self.activation == 'sigmoid':
                            o.sigmoid()
                        elif self.activation == 'logistic':
                            o.logistic()
                    for node in self.all_nodes:
                        node.input_values = []
                        node.sent = False
                    return self.softmax([val.itself_value for val in self.output_nodes])
                for node in self.h_plus_o:
                    if node.n_of_connections == len(node.input_values) and node.sent is False:
                        node.calculate_val()
                        for g in self.genome[1]:
                            if g.in_node == node.id and g.enabled is True:
                                try:
                                    out_node = self.all_nodes[self.nodes_lst.index(g.out_node)]
                                except:
                                    self.fix_net()
                                    out_node = self.all_nodes[self.nodes_lst.index(g.out_node)]
                                out_node.input_values.append(node.itself_value * g.weight)
                        node.sent = True
                all_inputs = sum([len(n.input_values) for n in self.h_plus_o])
            for o in self.output_nodes:
                o.calculate_val()
                if self.activation == 'sigmoid':
                    o.sigmoid()
                elif self.activation == 'logistic':
                    o.logistic()
            for node in self.all_nodes:
                node.input_values = []
                node.sent = False
            if self.softmax_bool:
                return self.softmax([val.itself_value for val in self.output_nodes])
            else:
                return [val.itself_value for val in self.output_nodes]
