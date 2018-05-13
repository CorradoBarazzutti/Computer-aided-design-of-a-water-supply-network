# -*- coding: utf-8 -*-
import string

import networkx as nx
import math
import numpy as np

import pandas as pd

from source.graphIO import graph_reader
from source.graphIO import display_graph

"""
Created on Mon Dec  4 22:57:30 2017

@author: Conrad
"""


class Router(object):
    # ----------------------------------------------------------------------------
    # -- CLASS ATTRIBUTES
    # ----------------------------------------------------------------------------

    # Class description
    CLASS_NAME = "Router"
    CLASS_AUTHOR = "Marcello Vaccarino"

    # ----------------------------------------------------------------------------
    # -- CLASS ATTRIBUTES
    # ----------------------------------------------------------------------------
    graph = nx.Graph()
    sinksource_graph = nx.Graph()
    acqueduct_1level = nx.Graph()
    acqueduct_2level = nx.Graph()
    acqueduct = nx.Graph()

    # ----------------------------------------------------------------------------
    # -- INITIALIZATION
    # ----------------------------------------------------------------------------
    def __init__(self, topo_file=None, building_file=None, adjacency_metrix=None):

        reader = graph_reader(self.graph)

        if topo_file != None and building_file != None:
            try:
                # [TODO] this function does not read building but single points
                reader.read_shp(topo_file, building_file)
            except Exception as e:
                raise e
        elif building_file != None:
            try:
                reader.read_shp_bilding(building_file)
            except Exception as e:
                raise e
        elif topo_file != None:
            try:
                reader.read_vtk(topo_file)
            except Exception as e:
                raise e
        elif adjacency_metrix != None:
            reader.read_adjacency(adjacency_metrix)

    # ----------------------------------------------------------------------------
    # -- CLASS METHODS
    # ----------------------------------------------------------------------------

    def write2shp(self, G, filename):
        try:
            import shapefile
        except ImportError:
            raise ImportError("read_shp requires pyshp")

        w = shapefile.Writer(shapeType=3)

        w.fields = [("DeletionFlag", "C", 1, 0), ["DC_ID", "N", 9, 0],
                    ["LENGHT", "N", 18, 5], ["NODE1", "N", 9, 1], ["NODE2", "N", 9, 0],
                    ["DIAMETER", "N", 18, 5], ["ROUGHNESS", "N", 18, 5], ["MINORLOSS", "N", 18, 5],
                    ["STATUS", "C", 1, 0]]

        i = 0
        for edge, _ in G.edges.items():
            node1 = [edge[0][0], edge[0][1]]
            node2 = [edge[1][0], edge[1][1]]
            line = [node1, node2]
            w.line(parts=[line])
            w.record(i, 1, 1, 2, 100, 0.1, 0, "1")
            i += 1
        w.save(filename)

    def write2epanet(self, G, filename):
        fo = open(filename + ".inp", "w")

        fo.write("[TITLE]\n")
        fo.write(filename)
        fo.write("\n\n")

        fo.write("[JUNCTIONS]\n")
        fo.write(";ID Elev Demand\n")
        for ID, node in enumerate(G.nodes.items()):
            node, datadict = node
            if datadict["Tank"] == False:
                fo.write(str(ID) + " " + str(round(datadict["ELEVATION"], 2)) + " " + str(datadict["DEMAND"]) + "\n")
        fo.write("\n")

        fo.write("[RESERVOIRS]\n")
        fo.write(";ID Head\n")
        fo.write("\n")

        fo.write("[TANKS]\n")
        fo.write(";ID Elev InitLvl MinLvl MaxLvl Diam Volume\n")
        for ID, node in enumerate(G.nodes.items()):
            node, datadict = node
            if datadict["Tank"] == True:
                fo.write(str(ID) + " " +
                         str(round(datadict["ELEVATION"], 2)) + " " +
                         str(round(datadict["ELEVATION"] + 5, 2)) + " " +
                         str(round(datadict["ELEVATION"] + 4, 2)) + " " +
                         str(round(datadict["ELEVATION"] + 30, 2)) + " " +
                         str(round(50, 2)) + " " + "\n")

        fo.write("\n")

        fo.write("[COORDINATES]\n")
        fo.write(";Node       X-Coord.    Y-Coord\n")
        for ID, node in enumerate(G.nodes.items()):
            node, datadict = node
            fo.write(str(ID) + " " + str(node[0]) + " " + str(node[1]) + "\n")
        fo.write("\n")

        fo.write("[PIPES]\n")
        fo.write(";ID Node1 Node2 Length Diam Roughness\n")
        for ID, edge in enumerate(G.edges.items()):
            edge, datadict = edge
            fo.write(str(ID) + " " + str(datadict["NODE1"])
                     + " " + str(datadict["NODE2"]) + " " + str(round(datadict["LENGHT"], 2))
                     + " " + str(datadict["DIAMETER"]) + " " + str(datadict["ROUGHNESS"]) + "\n")
        fo.write("\n")

        fo.write("[PUMPS]\n")
        fo.write(";ID Node1 Node2 Parameters\n")
        fo.write("\n")

        fo.write("[PATTERNS]\n")
        fo.write(";ID   Multipliers\n")
        fo.write("Pat1 " + "1" + "\n")

        fo.write("[CURVES]\n")
        fo.write(";ID  X-Value  Y-Value\n")
        fo.write("\n")

        fo.write("[QUALITY]\n")
        fo.write(";Node InitQual\n")
        fo.write("\n")

        fo.write("[REACTIONS]\n")
        fo.write("Global Bulk  -1\n")
        fo.write("Global Wall  0\n\n")

        fo.write("[TIMES]\n")
        fo.write("Hydraulic Timestep 1:00\n")
        fo.write("Pattern Timestep 6:00\n\n")

        '''
        [REPORT]
        Page      55
        Energy    Yes
        Nodes All
        Links All

        [OPTIONS]
        Units GPM
        Headloss H-W
        Pattern 1
        Quality Chlorine mg/L
        Tolerance 0.01
        '''

        fo.write("[END]")
        fo.close()

    def write2list(self, G, file_name):
        fo = open(file_name + ".txt", "w")
        for n1 in enumerate:
            pass
        fo.close()

    def write2vtk(self, G, filename):

        # import sys
        # sys.path = ['..'] + sys.path
        import pyvtk

        points = [list(node) for node, data in G.nodes(data=True)]
        line = []
        for edge in G.edges():
            for i, node in enumerate(G.nodes()):
                if node == edge[0]:
                    n1 = i
            for i, node in enumerate(G.nodes()):
                if node == edge[1]:
                    n2 = i
            line.append([n1, n2])

        vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(points, line=line))
        vtk.tofile(filename, 'ascii')

    def add_node_unique(self, new_node, new_attributes):
        """
        grants that the node added is unique with respect to the pos
        attribute equality relationship
        """
        for node in self.graph.nodes(True):
            if node[1]["pos"] == new_attributes["pos"]:
                return node[0]
        self.graph.add_node(new_node, new_attributes)
        return new_node

    def read_vtk(self, file_name):
        import numpy as np
        try:
            from mesh import Mesh
        except ImportError:
            raise ImportError("read_vtk requires pymesh")

        # initialize the vtk reader
        reader = Mesh()

        # read the vtk
        reader.ReadFromFileVtk(file_name)

        # add nodes to the graph
        for index, node in enumerate(reader.node_coord):
            self.graph.add_node(index, pos=reader.node_coord[index])

        '''
        chuncker
        Principe basé sur le stockage CSR ou CRS (Compressed Row Storage)
        dont voici une illustration :
        Soient six nœuds numérotes de 0 à 5 et quatre  ́el ́ements form ́es par
        les nœuds (0, 2, 3) pour l' el ement 0, (1, 2, 4) pour l'element,
        (0, 1, 3, 4) pour l’ ́el ́e- ment 2 et (1, 5) pour l’ ́el ́ement 3.
        Deux tableaux sont utilis ́es, l’un pour stocker de fa ̧con contigu ̈e
        les listes de nœuds qui composent les  ́el ́ements (table 1), l’autre
        pour indiquer la position, dans ce tableau, ou` commence chacune de
        ces listes (table 2).
        Ainsi, le chiffre 6 en position 2 dans le tableau p elem2node indique
        que le premier nœud de l’ ́el ́ement 2 se trouve en position 6 du
        tableau elem2node. La derni`ere valeur dans p elem2node correspond au
        nombre de cellules (la taille) du tableau elem2node.

        elem2node
        0 | 2 | 3 | 1 | 2 | 4 | 0 | 1 | 3 | 4 | 1 | 5
        1   2   3   4   5   6   7   8   9   10  11  12
        ^       ^           ^               ^       ^

        p_elem2node
        0 | 3 | 6 | 10 | 12
        1   2   3    4    4
        '''

        def chuncker(array, p_array):
            return [array[p_array[i]:p_array[i + 1]]
                    for i in range(len(p_array) - 1)]

        # given a cell returns the edges implicitely defined in it
        def pairwise(seq):
            return [seq[i:i + 2] for i in range(len(seq) - 2)] + \
                   [[seq[0], seq[len(seq) - 1]]]

        datas = np.asarray([data['pos']
                            for _, data in self.graph.nodes(data=True)])

        def distance3D(u, v, datas):
            xi = datas[u][0]
            yi = datas[u][1]
            zi = datas[u][2]
            xj = datas[v][0]
            yj = datas[v][1]
            zj = datas[v][2]
            return math.sqrt((xi - xj) * (xi - xj) +
                             (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj))

        # add edges to the graph
        for cell in chuncker(reader.elem2node, reader.p_elem2node):
            for u, v in pairwise(cell):
                if u not in self.graph[v]:
                    self.graph.add_edge(u, v, weight=distance3D(u, v, datas))

    def distance(self, nodei, nodej):
        xi = nodei[0]
        yi = nodei[1]
        xj = nodej[0]
        yj = nodej[1]
        if len(nodei) == 3 and len(nodej) == 3:
            zi = nodei[2]
            zj = nodej[2]
            return math.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) +
                             (zi - zj) * (zi - zj))
        return math.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj))

    def shortest_path(self, node1, node2):
        """
        Calculates the shortest path on self.graph.
        Path is a sequence of traversed nodes
        """
        try:
            path = nx.shortest_path(self.graph, source=node1, target=node2,
                                    weight="weight")
        except:
            pass
        return path

    def path_lenght(self, path):
        """
        given a path on the graph returns the lenght of the path in the
        unit the coordinats are expressed
        """
        if path is None:
            return float("inf")
        lenght = 0.0
        # given the path (list of node) returns the edges contained
        path_edges = [path[i:i + 2] for i in range(len(path) - 2)]
        # itereate to edges and calculate the weight
        for u, v in path_edges:
            lenght += self.distance(u, v)
        return lenght

    def TSP(self, cities):
        '''
        declaring the adjacency matrix
        T = numpy.empty(shape=(len(cities),len(cities)))
        for u, i in cities:
            for v, j in itertools(cities):
                uv_path = shortest_path(u, v)
                T[i][j] = path_lenght(shortest_path)

            start = timeit.default_timer() #start timer
            paths = []
            for combo in itertools.permutations(range(1,len(T[0]))):
                lenght = 0
                prev = 0
                path = []
                path += [0]
                for elem in combo:
                    lenght += T[prev][elem]
                    prev = elem
                    path += [elem]
                lenght += T[combo[len(combo)-1]][0]
                path += [0]
                paths.append((path, lenght))
            stop = timeit.default_timer() # stop timer
            time = stop - start
            return paths, time
        '''

    def is_sourcesink(self, node):
        '''given a node as in the networkx.Graph.nodes(data=1)
        returns 1 if the node is a sink or a source, 0 elsewhere'''
        if not node[1]['FID'] == '':
            return 1
        return 0

    def compute_source_matrix(self):
        for node in self.graph.nodes(data=1):
            if self.is_sourcesink(node):
                self.sinksource_graph.add_node(node[0], node[1])

        for n1 in self.sinksource_graph.nodes():
            for n2 in self.sinksource_graph.nodes():
                if n1 is not n2:
                    path = self.shortest_path(n1, n2)
                    if path is not None:
                        self.sinksource_graph.add_edge(n1, n2,
                                                       {'dist': self.path_lenght(path),
                                                        'path': path})

    def design_minimal_aqueduct(self, G):
        minimal = nx.minimum_spanning_tree(G, weight='dist')
        return minimal

    def display_recouvring_graph(self, G):
        path = []
        for edge in G.edges(data=True):
            path += edge[2]['path']
        self.display_path(path)

    def complete_graph(self, G):
        for n1 in G.nodes():
            for n2 in G.nodes():
                if n1 != n2:
                    # attributes = {'path': [n1, n2], 'dist': self.distance(n1,n2)}
                    G.add_edge(n1, n2)
                    G.edges[n1, n2]['dist'] = self.distance(n1, n2)
                    G.edges[n1, n2]['path'] = [n1, n2]

    def mesh_graph(self, G, weight):
        """complexity (len(G.nodes))^3"""
        distances = nx.get_edge_attributes(G, weight)

        # condition to create the gabriel relative neighbour graph
        def neighbors(p, q):
            for r in G.nodes:
                if r != q and r != p:
                    def dist(n1, n2):
                        if (n1, n2) in distances:
                            return distances[(n1, n2)]
                        else:
                            return distances[(n2, n1)]

                    if max(dist(p, r), dist(q, r)) < dist(p, q):
                        return False
            return True

        # connect graph
        gabriel_graph = nx.Graph()
        for n1 in G.nodes():
            for n2 in G.nodes():
                if n1 != n2:
                    if neighbors(n1, n2):
                        gabriel_graph.add_edge(n1, n2)
        return gabriel_graph

    def graphToEdgeMatrix(self, G):
        node_dict = {node: index for index, node in enumerate(G)}

        # Initialize Edge Matrix
        edgeMat = [[0 for x in range(len(G))] for y in range(len(G))]

        # For loop to set 0 or 1 ( diagonal elements are set to 1)
        for i, node in enumerate(G):
            tempNeighList = G.neighbors(node)
            for neighbor in tempNeighList:
                edgeMat[i][node_dict[neighbor]] = 1
            edgeMat[i][i] = 1

        return edgeMat

    def cluster(self, G):
        '''
        Finds the clusters
        '''
        # imports from a machine learning package skit-learn
        from sklearn.cluster import MeanShift, estimate_bandwidth

        # creates a array with the 2D coordinats for each node
        X = [[node[0], node[1]] for node in G.nodes()]
        # extimates the dimensions of single clusters
        bandwidth = estimate_bandwidth(X, quantile=0.1,
                                       random_state=0, n_jobs=1)
        # find clustes
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(X)

        # labels is an array indicating, for each node, the cluster number
        labels = {node: ms.labels_[i] for i, node in enumerate(G.nodes())}
        cluster_centers = [(node[0], node[1]) for node in ms.cluster_centers_]
        return (labels, cluster_centers)

    def print_atr(self, G):
        print("attributes")
        for node, attributes in G.nodes.items():
            print(attributes)
        print("\n")

    def design_aqueduct(self, LEVEL=0):
        # Costante di conversion dall'unità di misura dalla carta a metri
        CONVERSION = 6373044.737 * math.pi / 180

        # clustering
        labels, cluster_centers = self.cluster(self.graph)

        # --- ADDUCTION ---
        adduction = nx.Graph()
        for node in cluster_centers:
            adduction.add_node(node)

        self.complete_graph(adduction)
        adduction = self.mesh_graph(adduction, weight='dist')

        self.acqueduct.add_edges_from(adduction.edges())
        nx.set_node_attributes(self.acqueduct, -1, "CLUSTER")
        nx.set_node_attributes(self.acqueduct, False, "Tank")
        nx.set_edge_attributes(self.acqueduct, 1, "LEVEL")

        # --- DISTRIBUTION ---
        # initialize distribution graphs
        distribution = [nx.Graph() for cluster in cluster_centers]
        # add nodes to the respective cluster
        for node in labels:
            cluster = labels[node]
            distribution[cluster].add_node(node)
        # add center to the respective cluster
        for cluster, center in enumerate(cluster_centers):
            distribution[cluster].add_node(center)

        # for each distribution sub-network
        for dist_graph in distribution:
            # connect nodes in the graph
            self.complete_graph(dist_graph)
            dist_graph = self.mesh_graph(dist_graph, weight='dist')
            # mark as distribution
            nx.set_edge_attributes(dist_graph, 2, "LEVEL")
            # add to acqueduct
            self.acqueduct.add_edges_from(dist_graph.edges())

        # add label info to the graph
        nx.set_node_attributes(self.acqueduct, labels, 'CLUSTER')

        # add elevation
        elevdict = nx.get_node_attributes(self.graph, "mean")
        nx.set_node_attributes(self.acqueduct, elevdict, "ELEVATION")
        moy = 0
        for center in adduction.nodes:
            for nbr in self.acqueduct.neighbors(center):
                if "ELEVATION" in self.acqueduct.nodes[nbr]:
                    moy += self.acqueduct.nodes[nbr]["ELEVATION"]
            moy = moy / len(list(self.acqueduct.neighbors(center)))
            self.acqueduct.nodes[center]["ELEVATION"] = moy
            self.acqueduct.nodes[center]["DEMAND"] = 0

        # add tank info to the graph
        namedict = nx.get_node_attributes(self.graph, "name")
        tankdict = dict([(node, True if namedict[node] == "tank" else False) for node in namedict])
        nx.set_node_attributes(self.acqueduct, tankdict, 'Tank')
        initial_level = 3
        nx.set_node_attributes(self.acqueduct, dict([(node, initial_level)
                                                     for node in tankdict if tankdict[node]]), "Linit")

        # add level
        for dist_graph in distribution:
            leveldict = nx.get_node_attributes(dist_graph, "LEVEL")
            nx.set_node_attributes(self.acqueduct, leveldict, "LEVEL")

        # add node id
        for ID, node in enumerate(self.acqueduct.nodes.items()):
            node, datadict = node
            nx.set_node_attributes(self.acqueduct, {node: ID}, "ID")

        # add demande
        building_type = nx.get_node_attributes(self.graph, "building")
        for ID, node in enumerate(self.acqueduct.nodes.items()):
            node, datadict = node
            building2demand = {"appartment": 0.11}
            default = 0.11
            demand = building2demand.get(building_type.get(node, ''), default)
            nx.set_node_attributes(self.acqueduct, {node: demand}, "DEMAND")

        # add edge attributes
        IDdict = nx.get_node_attributes(self.acqueduct, "ID")
        for ID, edge in enumerate(self.acqueduct.edges.items()):
            edge, datadict = edge
            node1 = edge[0]
            node2 = edge[1]
            datadict["DC_ID"] = ID
            datadict["LENGHT"] = self.distance(node1, node2) * CONVERSION
            datadict["NODE1"] = IDdict[node1]
            datadict["NODE2"] = IDdict[node2]
            datadict["DIAMETER"] = 500 if "LEVEL" in self.acqueduct[node1][node2] else 100
            datadict["ROUGHNESS"] = 120 if datadict["DIAMETER"] <= 200 else 130
            datadict["MINORLOSS"] = 0
            datadict["STATUS"] = 1
            datadict["LEVEL"] = 1 if edge in adduction.edges else 2
            for key in datadict:
                nx.set_edge_attributes(self.acqueduct, {edge: datadict[key]}, key)

        # case we only want the first level network
        if LEVEL == 1:
            adduction = self.acqueduct.copy()
            distribution = [node for node, _ in self.acqueduct.nodes.items() if not node in cluster_centers]
            adduction.remove_nodes_from(distribution)
            for cluster, center in enumerate(cluster_centers):
                self.acqueduct.nodes[center]["DEMAND"] = 0
                for node in labels:
                    if labels[node] == cluster:
                        adduction.nodes[center]["DEMAND"] += self.acqueduct.nodes[node]["DEMAND"]
            self.acqueduct = adduction

    def louvain_clustering(self, G, weight=None):
        # Automatic Partitioning of Water Distribution Networks Using Multiscale Community Detection and Multiobjective...
        def GN_modularity(G, s):
            # adjacency matrix
            A = nx.adjacency_matrix(G, weight=weight).toarray()
            # array of ones
            I = np.reshape(np.ones(len(A)), (-1, 1))
            # 
            D = np.matmul(A, I)
            # dirty to list
            A = A.tolist()
            D = [e[0] for e in D.tolist()]
            # m = |E| number of edges
            m = len(G.edges)

            # δ is the Kronecker delta function δ(i, j) = 1, if node i and j belongs to the same community C,
            # 0 otherwise
            def delta(i, j):
                return 1 if s[i] == s[j] else 0

            # double sum over matrix
            def mds(M):
                sumM = 0
                for row in M:
                    for e in row:
                        sumM += e
                return sumM

            # formula (2)
            Mat = [[(A[i][j] - ((k_i * k_j) / (2 * m))) * delta(i, j)
                    for j, k_j in enumerate(D)] for i, k_i in enumerate(D)]
            return 0.5 / m * mds(Mat)

        #
        def inner_loop(G, s):
            s_in = s.copy()
            # ---- Initialisation ----
            Q = GN_modularity(G, s_in)
            nextQ = Q
            # adjacency matrix
            A = nx.adjacency_matrix(G, weight=weight).toarray()
            # ---- Do - While ---- 
            while True:
                for n1, c1 in enumerate(s_in):
                    neighbour_comunitis = [c2 for n2, c2 in enumerate(s_in) if A[n1][n2] != 0]
                    for c2 in neighbour_comunitis:
                        tempS = s_in.copy()
                        tempS[n1] = c2
                        tempQ = GN_modularity(G, tempS)
                        if tempQ > nextQ:
                            nextS = tempS.copy()
                            nextQ = tempQ
                if nextQ <= Q:
                    break
                else:
                    s_in = nextS.copy()
                    Q = nextQ
            return s_in.copy()

        #
        def collapse(G, s):
            # verify data coherence
            if len(G.nodes) != len(s):
                raise Exeption("")
            # initialise graph 
            H = nx.Graph()
            # add nodes
            for super_node in set(s):
                H.add_node(super_node, nodes=[node for node, _ in enumerate(s) if s[node] == super_node])
            # add weighted edges
            G_nodes = {i: coord for i, coord in enumerate(G.nodes)}
            for super_node1, super_datadict1 in H.nodes.items():
                for i in super_datadict1["nodes"]:
                    n1 = G_nodes[i]
                    for super_node2, super_datadict2 in H.nodes.items():
                        for j in super_datadict2["nodes"]:
                            n2 = G_nodes[j]
                            if n2 in G[n1]:
                                if super_node2 in H[super_node1]:
                                    H[super_node1][super_node2]['weight'] += G[n1][n2]["weight"]
                                else:
                                    H.add_edge(super_node1, super_node1, weight=G[n1][n2]["weight"])
            return H

        #
        def inflate(H, G, sH):
            if len(sH) != len(H.nodes):
                raise ValueError('label array lenght does not mach graph H')
            sG = [0 for e in G.nodes]
            for super_node_index, super_node in enumerate(H.nodes.items()):
                super_node, super_datadict = super_node
                for node in super_datadict["nodes"]:
                    sG[node] = sH[super_node_index]
            return sG

        # ------------------------
        # ---- Initialisation ----
        s = [i for i, _ in enumerate(G.nodes)]
        Q = GN_modularity(G, s)
        nextQ = Q
        # ---- Do - While ----
        while True:
            tempS = inner_loop(G, s)
            H = collapse(G, tempS)
            sH = [i for i, _ in enumerate(H.nodes)]
            sH = inner_loop(H, sH)
            tempS = inflate(H, G, sH)
            tempQ = GN_modularity(G, tempS)
            if tempQ > nextQ:
                nextS = tempS.copy()
                nextQ = tempQ
            if nextQ <= Q:
                break
            else:
                s = nextS.copy()
                Q = nextQ
        return s

    def route_vesuvio(self, n1, n2):
        try:
            import shapefile
        except ImportError:
            raise ImportError("read_shp requires pyshp")
        # route
        path = self.shortest_path(n1, n2)

        # turn path into acqueduct graph
        datas = [data['pos'] for _, data in self.graph.nodes(data=True)]
        path_coord = [tuple(datas[node]) for node in path]
        path_edges = [path_coord[i:i + 2] for i in range(len(path_coord) - 2)]
        self.acqueduct.add_edges_from(path_edges)

        # write shp
        def write2shape():
            w = shapefile.Writer(shapeType=3)
            w.field("name", "C")
            line = path_edges
            w.line(parts=line)
            w.record('path')
            w.save('path')

    """ 
    def hardy_cross(self):
        alphabet = list(string.ascii_lowercase)
        node2letter = {key: value for (key, value) in zip(self.acqueduct.nodes, alphabet)}
        letter2node = {key: value for (key, value) in zip(alphabet, self.acqueduct.nodes)}
        cycles = [[node2letter[node] for node in cycle] for cycle in nx.cycle_basis(self.acqueduct)]

        def pairwise(it):
            return [it[i] + it[i + 1] for i in range(len(it) - 1)] + [it[len(it) - 1] + it[0]]

        cycles = [pairwise(cycle) for cycle in cycles]

        # node data dictionaries
        datadicts = {node2letter[node[0]] + node2letter[node[1]]: datadict
                     for (node, datadict) in self.acqueduct.edges.items()}

        def get_att(dict, key):
            if key in dict:
                return dict[key]
            else:
                return dict[key[1] + key[0]]

        '''---- Initial guess ---
        kirchoff low on nodes to find flow on edges'''
        import z3
        solver = z3.Solver()
        X = dict([(node, z3.Real("x_%s" % i)) for i, (node, datadict) in enumerate(self.acqueduct.nodes.items())])
        Y = dict([(edge, z3.Real("y_%s" % i)) for i, (edge, datadict) in enumerate(self.acqueduct.edges.items())])
        # boundary conditions
        boudary_c = []
        for node, datadict in self.acqueduct.nodes.items():
            if not datadict["Tank"]:
                boudary_c.append(X[node] == datadict["DEMAND"])
        # boudary_c = [X[node] == datadict["DEMAND"] for node, datadict in self.acqueduct.nodes.items()]
        # kirchoff lows
        edges = set([node2letter[edge[0]] + node2letter[edge[1]] for edge in self.acqueduct.edges])

        def is_edge(e):
            return e in edges

        def is_antiedge(e):
            return e[1] + e[0] in edges

        def get_data(dict, edge):
            if is_edge(edge):
                return dict[edge]
            elif is_antiedge(edge):
                return -dict[edge[1] + edge[0]]

        kirchoff_c = [z3.Sum([X[n1]] + [Y[(n1, n2)] if (n1, n2) in Y else -Y[(n2, n1)]
                                        for n2 in self.acqueduct.neighbors(n1)]) == 0
                      for n1 in self.acqueduct.nodes]
        solver.add(boudary_c + kirchoff_c)
        if solver.check() == z3.sat:
            m = solver.model()
            guesses = [float(m.evaluate(Y[edge]).numerator_as_long()) / float(m.evaluate(Y[edge]).denominator_as_long())
                       for edge in Y]
        else:
            print("failed to solve")
        manual_guess = {'eg': ((+ 47.19 - 3.629999999999999) / 2 - 6.600000000000006),
                        'ad': (+ 47.19 - 3.629999999999999) / 2 - 12.429999999999987 - 9.570000000000002,
                        'ce': (47.19 - 3.629999999999999) / 2,
                        'dg': (
                                          + 47.19 - 3.629999999999999) / 2 - 12.429999999999987 - 9.570000000000002 - 3.2999999999999994,
                        'cf': - 47.19,
                        'gh': 7.480000000000008,
                        'ab': 9.570000000000002,
                        'ac': (- 47.19 + 3.629999999999999) / 2}
        manual_array = {}
        for key in manual_guess:
            for cycle in cycles:
                if key in cycle or key[1] + key[0] in cycle:
                    manual_array[key] = manual_guess[key]

        # convert data format
        loops = []
        for cycle in cycles:
            loops.append(pd.DataFrame(data={key: value
                                            for (key, value) in [
                                                # ("Section", np.array(cycle)),
                                                ("Section", np.array(list(manual_array.keys()))),
                                                ("L", np.array([get_att(datadicts, edge)["LENGHT"] for edge in cycle])),
                                                # ("Q", np.array([get_data(dict(zip(edges, guesses)),edge) for edge in cycle])),
                                                ("Q", np.array(list(manual_array.values()))),
                                                ("D", np.zeros(len(cycle), float)),
                                                ("J", np.zeros(len(cycle), float)),
                                                ("hf", np.zeros(len(cycle), float)),
                                                ("hf/Q", np.zeros(len(cycle), float))]}))
        # ---- Hardy Cross ----
        hc_solver = hc.HardyCross(loops)
        hc_solver.sort_edge_names()
        hc_solver.locate_common_loops()
        hc_solver.run_hc()
        datadict = dict([(edge, datadict) for edge, datadict in self.acqueduct.edges.items()])

        def set_att(dict, edge, key, value):
            if edge in dict:
                dict[edge][key] = value
                return True
            elif (edge[1], edge[0]) in dict:
                dict[(edge[1], edge[0])] = value

        def set_att_sign(dict, edge, key, value):
            if edge in dict:
                dict[edge][key] = value
                return True
            elif (edge[1], edge[0]) in dict:
                dict[(edge[1], edge[0])] = - value

        for loop in hc_solver.loops:
            for (section, diameter) in zip(loop['Section'], loop["D"]):
                set_att(datadict, (letter2node[section[0]], letter2node[section[1]]), "DIAMETER", diameter)
            for (section, diameter) in zip(loop['Section'], loop["Q"]):
                set_att_sign(datadict, (letter2node[section[0]], letter2node[section[1]]), "Q", diameter)
            for (section, diameter) in zip(loop['Section'], loop["hf"]):
                set_att_sign(datadict, (letter2node[section[0]], letter2node[section[1]]), "hf", diameter)
        # set flow attribute
        nx.set_edge_attributes(self.acqueduct, dict([(edge, Q) for edge, Q in zip(self.acqueduct.edges, guesses)]), "Q")
        nx.set_edge_attributes(self.acqueduct,
                               dict([(edge, datadict[edge]["Q"]) for edge in datadict]),
                               "Q")
        # set diameter
        diameters = dict(
            [(edge, hc.diameter_from_available(np.sqrt((4 * np.abs(Q) * 10 ** -3) / (np.pi * 1)) * 10 ** 3))
             for edge, Q in nx.get_edge_attributes(self.acqueduct, "Q").items()])
        nx.set_edge_attributes(self.acqueduct, diameters, "DIAMETER")
        nx.set_edge_attributes(self.acqueduct,
                               dict([(edge, datadict[edge]["DIAMETER"]) for edge in datadict]),
                               "DIAMETER")

        '''---- Kirchoff lows on loops ----
        in order to find heads and pressure in each node'''
        solver_head = z3.Solver()
        H = dict([(node, z3.Real("h_%s" % node2letter[node])) for i, (node, datadict) in
                  enumerate(self.acqueduct.nodes.items())])
        # boundary condition
        tank_c = []
        for node, datadict in self.acqueduct.nodes.items():
            if datadict["Tank"] == True:
                tank_c.append(H[node] == datadict["ELEVATION"] + datadict["Linit"])

        kirchoff_loops_c = []
        for edge, datadict in self.acqueduct.edges.items():
            if "hf" in datadict:
                kirchoff_loops_c += [z3.And(H[edge[1]] - H[edge[0]] >= datadict["hf"] * 0.9,
                                            H[edge[1]] - H[edge[0]] <= datadict["hf"] * 1.1)]
            else:
                deltaH = datadict["LENGHT"] * (datadict["Q"] * abs(datadict["Q"])) / (
                        math.sqrt(abs(datadict["Q"]) * 4 / math.pi) ** 5.33)
                kirchoff_loops_c += [z3.And(H[edge[1]] - H[edge[0]] >= deltaH * 0.9,
                                            H[edge[1]] - H[edge[0]] <= deltaH * 1.1)]
        solver_head.add(tank_c + kirchoff_loops_c)
        if solver_head.check() == z3.sat:
            m = solver.model()
            head = [float(m.evaluate(H[node]).numerator_as_long()) / float(m.evaluate(H[node]).denominator_as_long())
                    for node in H]
        else:
            print("failed to solve")
    """

    def solve(self, G):

        import z3
        # add a solver
        solver = z3.Solver()

        # +-----------------------+
        # | variables declaration |
        # +-----------------------+

        # water demand in each node
        X = dict([(node, z3.Real("demand_%s" % i))
                  for i, (node, datadict) in enumerate(G.nodes.items())])
        # water flow in each pipe
        Q = dict([(edge, z3.Real("Q_%s" % i))
                  for i, (edge, datadict) in enumerate(G.edges.items())])
        # water speed in each pipe
        V = dict([(edge, z3.Real("V_%s" % i))
                  for i, (edge, datadict) in enumerate(G.edges.items())])
        # pipes diameter
        D = dict([(edge, z3.Real("D_%s" % i))
                  for i, (edge, datadict) in enumerate(G.edges.items())])

        # +---------------------+
        # | boundary conditions |
        # +---------------------+

        # water demand in each node
        boudary_c = []
        for node, datadict in G.nodes.items():
            if not datadict["Tank"]:
                boudary_c.append(X[node] == datadict["DEMAND"])

        # closing equation in each pipe: Q = V * A
        def sym_abs(x):
            return z3.If(x >= 0, x, -x)

        closing_c = [sym_abs(Q[edge]) / 1000 == V[edge] * (D[edge] / 1000 / 2) ** 2 * math.pi
               for edge in G.edges]

        # speed limits
        speed_c = []
        for edge, datadict in G.edges.items():
            if datadict["LEVEL"] == 1:
                speed_c.append(z3.And(V[edge] >= 0.5, V[edge] <= 1.))
            else:
                speed_c.append(z3.And(V[edge] >= 0.3, V[edge] <= 1.5))

        # pipes diameters
        diameter_c = []
        available_measures = [75., 90., 110., 125., 140., 160., 180., 200., 225., 250., 280., 315., 355., 400.]
        for edge, datadict in G.edges.items():
            if datadict["LEVEL"] == 1:
                diameter_c += [z3.Or([D[edge] == measure for measure in available_measures])]
            else:
                diameter_c += [z3.Or([D[edge] == measure for measure in available_measures + [15., 25., 50.]])]

        # kirchoff lows in the nodes
        kirchoff_c = []
        for n1 in G.nodes:
            kirchoff_c.append(z3.Sum([X[n1]] + [Q[(n1, n2)] if (n1, n2) in Q else -Q[(n2, n1)]
                                                for n2 in G.neighbors(n1)]) == 0)
        # solve theory
        solver.add(boudary_c + closing_c + speed_c + kirchoff_c + diameter_c)
        if solver.check() == z3.sat:
            m = solver.model()
            X = dict([(node, float(m.evaluate(X[node]).numerator_as_long()) / float(
                m.evaluate(X[node]).denominator_as_long()))
                      for node in X])
            nx.set_node_attributes(G, X, "Q")
            Q = dict([(edge, float(m.evaluate(Q[edge]).numerator_as_long()) / float(m.evaluate(Q[edge]).denominator_as_long()))
                       for edge in Q])
            V = dict([(edge, float(m.evaluate(V[edge]).numerator_as_long()) / float(m.evaluate(V[edge]).denominator_as_long()))
                       for edge in V])
            D = dict([(edge, float(m.evaluate(D[edge]).numerator_as_long()) / float(m.evaluate(D[edge]).denominator_as_long()))
                       for edge in D])
            for key, datadict in [("Q", Q), ("V", V), ("DIAMETER", D)]:
                nx.set_edge_attributes(G, datadict, key)
            print("solved")
        else:
            print("failed to solve")

        for node, datadict in G.nodes.items():
            if datadict["Tank"] == True:
                start = node
        H = {start: G.nodes[start]["ELEVATION"] + 56}
        visited, queue = set(), [start]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(set(G.neighbors(node)) - visited)
                for neighbour in G.neighbors(node):
                    L = G[node][neighbour]["LENGHT"]
                    K = 10.29 / 70**2
                    Q = G[node][neighbour]["Q"]
                    D = G[node][neighbour]["DIAMETER"]
                    H[neighbour] = H[node] - (L * K * (Q * abs(Q)) / 1000000 / (D / 1000) ** 5.33)

        nx.set_node_attributes(G, H, "H")