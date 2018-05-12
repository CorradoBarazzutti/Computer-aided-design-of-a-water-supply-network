import networkx as nx

class kpi_calculator():

    wdn = nx.Graph()

    # Available Power (or total power)
    available_power = 0
    resilience = 0

    MIN = 0
    MAX = 0
    MEAN = 0
    SQM = 0

    def __init__(self, wdn):
        self.wdn = wdn
        self.statistical_idexs()
        self.energy_indexs()

    def energy_indexs(self):
        sum = 0
        for node, datadict in self.wdn.nodes.items():
            if datadict["Tank"] == True:
                # [m] * 1000 * [l] * [kg / l^3 / s]
                sum += datadict["Q"] * 1000 * datadict["hf"]
        self.availble_power = sum

    def statistical_idexs(self):
        MIN = float("+inf")
        MAX = float("-inf")
        MEAN = 0
        MENQ = 0
        for node, datadict in self.wdn.nodes.items():
            H = datadict["hf"]
            MEAN += H
            MEAQ += H**2
            if H < MIN:
                MIN = H
            if H > MAX:
                MAX = H
        self.MIN = MIN
        self.MAX = MAX
        self.MEAN = MEAN / len(self.wdn.nodes)
        self.SQM = MENQ / len(self.wdn.nodes) - MEAN ** 2
