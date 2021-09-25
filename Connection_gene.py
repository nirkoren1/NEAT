import numpy as np
from innov import InnovOb

innov_marker = [-1, -1, -1, -1]
low_weight = 0
high_weight = 1


def get_innov_num(in_n, out_n, innov_lst, index):
    global innov_marker
    if len(innov_lst) == 0:
        innov_marker[index] += 1
        innov_lst.append(InnovOb(innov_marker[index], in_n, out_n))
        return innov_marker[index]
    else:
        for innov_indx in range(len(innov_lst)):
            innov = innov_lst[innov_indx]
            if innov.in_node == in_n and innov.out_node == out_n:
                return innov.id
        innov_marker[index] += 1
        innov_lst.append(InnovOb(innov_marker[index], in_node=in_n, out_node=out_n))
        return innov_marker[index]


class ConGene:
    def __init__(self, innov_lst=None, existing_gene=None, in_node=0, out_node=0, index=0):
        if existing_gene is None:
            if in_node == 0 or out_node == 0:
                print("you must enter in_node and out_node")
                raise ValueError
            self.in_node = in_node
            self.out_node = out_node
            self.weight = np.random.uniform(low_weight, high_weight)
            self.enabled = True
            self.innov = get_innov_num(self.in_node, self.out_node, innov_lst=innov_lst, index=index)
        else:
            self.in_node = existing_gene.in_node
            self.out_node = existing_gene.out_node
            self.weight = existing_gene.weight
            self.enabled = existing_gene.enabled
            self.innov = existing_gene.innov
