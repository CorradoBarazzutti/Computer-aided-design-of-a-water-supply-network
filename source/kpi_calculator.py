from typing import List, Any

import networkx as nx

class kpi_calculator():

    wdn = nx.Graph()

    # Available Power (or total power):
    available_power = 0
    # Dissipated Power (or internal power): sum (􏰀qj * ∆Hj) over each edge
    dissipated_power = 0
    # Resilience index: Ir = 1 − P_D / P_Dmax
    resilience = 0

    MIN = 0
    MAX = 0
    MEAN = 0
    SQM = 0

    def __init__(self, wdn):
        self.wdn = wdn

        # energy
        tank_list = []
        s = 0
        for node, datadict in self.wdn.nodes.items():
            if datadict["Tank"] == True:
                tank_list.append(node)
                s += abs(datadict["Q"]) * datadict["H"]
        self.available_power = s

        portate = dict()
        for edge, data in nx.get_edge_attributes(self.wdn, "Q").items():
            if not ((tuple(edge[0]) in tank_list) or (tuple(edge[1]) in tank_list)):
                portate[edge] = data
        headdict = dict()
        for node, data in nx.get_node_attributes(self.wdn, "H").items():
            if not (tuple(node) in tank_list):
                headdict[node] = data
        caduta = [abs(headdict[edge[0]] - headdict[edge[1]]) for edge in portate]
        a = map(lambda t: abs(t[0] * t[1]), zip(portate.values(), caduta))
        self.dissipated_power = sum(a)

        demand = nx.get_node_attributes(self.wdn, "Q").values()
        head = nx.get_node_attributes(self.wdn, "H").values()
        a = map(lambda t: abs(t[0] * t[1]), zip(demand, head))
        self.nodes_power = sum(a) - self.available_power

        self.resilience = 1 - self.dissipated_power / (self.available_power - self.nodes_power)

        # statistical
        MIN = float("+inf")
        MAX = float("-inf")
        MEAN = 0
        MEAQ = 0
        for node, datadict in self.wdn.nodes.items():
            H = datadict["H"]
            MEAN += H
            MEAQ += H**2
            if H < MIN:
                MIN = H
            if H > MAX:
                MAX = H
        self.MIN = MIN
        self.MAX = MAX
        self.MEAN = MEAN / len(self.wdn.nodes)
        self.SQM = MEAQ / len(self.wdn.nodes) - MEAN ** 2

        self.print_kpi()

    def print_kpi(self):
        print("available_power %s" % self.available_power)
        print("dissipated_power %s" % self.dissipated_power)
        print("nodes_power %s" % self.nodes_power)
        print("resilience %s" % self.resilience)

        print("MIN %s" % self.MIN)
        print("MAX %s" % self.MAX)
        print("MEAN %s" % self.MEAN)
        print("SQM %s" % self.SQM)
