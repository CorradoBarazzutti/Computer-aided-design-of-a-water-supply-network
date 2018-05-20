import math
from typing import List, Any

import networkx as nx

class kpi_calculator():
    """
    This class computes performance indicators for a solved water distribuion network.
    Varius types of indicators are available:
    - Energy indicators of the network
        - available_power
        - dissipated_power
    - Statistical indicators of nodes head per cluster
        - min
        - max
        - mean
        - mean square error
    """
    wdn = nx.Graph()

    # Available Power (or total power):
    available_power = 0
    # Dissipated Power (or internal power): sum (􏰀qj * ∆Hj) over each edge
    dissipated_power = 0

    MIN = []
    MAX = []
    MEAN = []
    SQM = []

    def __init__(self, wdn):
        self.wdn = wdn

        # energy
        tank_list = []
        s = 0

        for node, datadict in self.wdn.nodes.items():
            if datadict["Tank"] == True:
                tank_list.append(node)
                s += abs(2 * datadict["DEMAND"]) * datadict["H"]
        self.available_power = s * 9.81

        portate = nx.get_edge_attributes(self.wdn, "Q")
        headdict = nx.get_node_attributes(self.wdn, "H")
        caduta = [abs(headdict[edge[0]] - headdict[edge[1]]) for edge in portate]
        a = map(lambda t: abs(t[0] * t[1]), zip(portate.values(), caduta))
        self.dissipated_power = sum(a) * 9.81

        demand = nx.get_node_attributes(self.wdn, "Q").values()
        head = nx.get_node_attributes(self.wdn, "H").values()
        a = map(lambda t: abs(t[0] * t[1]), zip(demand, head))
        self.nodes_power = sum(a) * 9.81 - self.available_power

        # statistical
        # get number of different clusters
        nc = max(nx.get_node_attributes(self.wdn, 'CLUSTER').values()) + 2
        print(set(nx.get_node_attributes(self.wdn, 'CLUSTER').values()))
        # nodes per clusters
        npc = [0 for i in range(nc)]
        MIN = [float("+inf") for i in range(nc)]
        MAX = [float("-inf") for i in range(nc)]
        MEAN = [0 for i in range(nc)]
        MEAQ = [0 for i in range(nc)]

        for node, datadict in self.wdn.nodes.items():
            i = datadict["CLUSTER"] + 1
            npc[i] += 1
            H = datadict["H"] - datadict["ELEVATION"]
            MEAN[i] += H
            MEAQ[i] += H**2
            if H < MIN[i]:
                MIN[i] = H
            if H > MAX[i]:
                MAX[i] = H

        self.MIN = MIN
        self.MAX = MAX
        for i in range(nc):
            self.MEAN.append(MEAN[i] / npc[i])
            self.SQM.append(math.sqrt(MEAQ[i] / npc[i] - self.MEAN[i] ** 2))

        self.clusters_pression = [datadict["H"] - datadict["ELEVATION"]
                                  for node, datadict in self.wdn.nodes.items() if datadict["CLUSTER"] == -1]

        self.print_kpi()

    def print_kpi(self):
        print("available_power %s" % self.available_power)
        print("dissipated_power %s" % self.dissipated_power)
        print("nodes_power %s" % self.nodes_power)

        print("MIN %s" % self.MIN)
        print("MAX %s" % self.MAX)
        print("MEAN %s" % self.MEAN)
        print("SQM %s" % self.SQM)

        print("Centers pressions = %s" % self.clusters_pression)
