# -*- coding: utf-8 -*-

import networkx as nx
import math
import sys

reload(sys)
sys.setdefaultencoding('utf8')

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

    # Attributes
    graph = nx.Graph()
    sinksource_graph = nx.Graph()
    acqueduct = nx.Graph()
    # ----------------------------------------------------------------------------
    # -- INITIALIZATION
    # ----------------------------------------------------------------------------

    def __init__(self, topo_file=None, building_file=None):
        # self.read_shp(topo_file, point_file)
        # self.compute_source_matrix()
        self.read_shp_bilding(building_file)

    # ----------------------------------------------------------------------------
    # -- CLASS ATTRIBUTES
    # ----------------------------------------------------------------------------
    def avg(self, node_list):
        x = 0
        y = 0
        for node in node_list:
            x += node[0]
            y += node[1]
        x /= len(node_list)
        y /= len(node_list)
        return (x, y)

    def read_shp_bilding(self, file_name):

        try:
            import shapefile
        except ImportError:
            raise ImportError("read_shp requires pyshp")

        sf = shapefile.Reader(file_name)
        fields = [x[0] for x in sf.fields]

        for shapeRecs in sf.iterShapeRecords():
            shape = shapeRecs.shape
            for cell in self.row_chuncker(shape.points, shape.parts):
                center_coord = self.avg(cell)
                attributes = dict(zip(fields, shapeRecs.record))
                attributes['pos'] = center_coord
                self.graph.add_node(center_coord, attributes)

    # chuncker: see commpresed row storage
    def row_chuncker(self, array, p_array):
        p_array.append(len(array))
        return [array[p_array[i]:p_array[i+1]]
                for i in range(len(p_array)-1)]

    def read_shp(self, file_name, point_file=None):
        """Generates a networkx.DiGraph from shapefiles. Point geometries are
        translated into nodes, lines into edges. Coordinate tuples are used as
        keys. Attributes are preserved, line geometries are simplified into
        start and end coordinates. Accepts a single shapefile or directory of
        many shapefiles.

        "The Esri Shapefile or simply a shapefile is a popular geospatial
        vector data format for geographic information systems software."

        Parameters
        ----------
        path : file or string
           File, directory, or filename to read.

        simplify:  bool
            If ``True``, simplify line geometries to start and end coordinates.
            If ``False``, and line feature geometry has multiple segments, the
            non-geometric attributes for that feature will be repeated for each
            edge comprising that feature.

        Returns
        -------
        G : NetworkX graph

        Examples
        --------
        >>> G=nx.read_shp('test.shp') # doctest: +SKIP

        References
        ----------
        .. [1] http://en.wikipedia.org/wiki/Shapefile
        """
        try:
            import shapefile
        except ImportError:
            raise ImportError("read_shp requires pyshp")

        sf = shapefile.Reader(file_name)
        fields = [x[0] for x in sf.fields]

        for shapeRecs in sf.iterShapeRecords():

            shape = shapeRecs.shape
            if shape.shapeType == 1:    # point
                attributes = dict(zip(fields, shapeRecs.record))
                attributes["pos"] = shape.points[0]
                self.graph.add_node(shape.points[0], attributes)

            if shape.shapeType == 3:    # polylines
                attributes1 = dict(zip(fields, shapeRecs.record))
                attributes2 = dict(zip(fields, shapeRecs.record))
                for i in range(len(shape.points) - 1):
                    '''
                    attributes1["pos"] = shape.points[i]
                    n1 = self.add_node_unique(shape.points[i], attributes1)
                    attributes2["pos"] = shape.points[i + 1]
                    n2 = self.add_node_unique(shape.points[i + 1], attributes2)
                    attribute = {'dist': self.distance(n1, n2)}
                    print '{0}: {1}, {2}'.format(i, n1, n2)
                    self.graph.add_edge(n1, n2, attribute) '''
                    attributes1["pos"] = shape.points[i]
                    n1 = shape.points[i]
                    self.graph.add_node(n1, attributes1)
                    attributes2["pos"] = shape.points[i + 1]
                    n2 = shape.points[i + 1]
                    self.graph.add_node(n2, attributes2)
                    attribute = {'dist': self.distance(n1, n2)}
                    self.graph.add_edge(n1, n2, attribute)

            if shape.shapeType == 5:    # polygraph
                # chuncker: see commpresed row storage
                def chuncker(array, p_array):
                    p_array.append(len(array))
                    return [array[p_array[i]:p_array[i+1]]
                            for i in range(len(p_array)-1)]

                # given a cell returns the edges (node touple) implicitely defined in it
                def pairwise(seq):
                    return [seq[i:i+2] for i in range(len(seq)-1)]

                for cell in chuncker(shape.points, shape.parts):
                    for n1, n2 in pairwise(cell):
                        attributes1 = dict(zip(fields, shapeRecs.record))
                        attributes1["pos"] = n1
                        attributes2 = dict(zip(fields, shapeRecs.record))
                        attributes2["pos"] = n2
                        # add nodes of the shape to the graph
                        n1 = self.add_node_unique(n1, attributes1)
                        n2 = self.add_node_unique(n2, attributes2)
                        attribute = {'dist': self.distance(n1, n2)}
                        # add edge
                        self.graph.add_edge(n1, n2, attribute)

        if point_file != None:
            sf = shapefile.Reader(point_file)
            new_fields = [x[0] for x in sf.fields]
            nodes2attributes = {node: data \
                                for node, data in self.graph.nodes(data=1)}
            for shapeRecs in sf.iterShapeRecords():
                shape = shapeRecs.shape
                if shape.shapeType != 1:    # point
                    raise ValueError("point_file must be of type 1: points")
                new_attributes = dict(zip(new_fields, shapeRecs.record))
                nodes = [tuple(point) for point in shape.points]
                for node in nodes:
                    # print node in nodes2attributes, new_attributes
                    nodes2attributes[node].update(new_attributes)
            for node, data in self.graph.nodes(data=1):
                data.update(nodes2attributes[node])
            # nx.set_node_attributes(self.graph, nodes2attributes)

    def write2shp(self):
        try:
            import shapefile
        except ImportError:
            raise ImportError("read_shp requires pyshp")

        w = shapefile.Writer(shapeType=3)
        w.field("name", "C")
        
        for edge in self.acqueduct.edges():
            line = [edge[0], edge[1]]
            w.line(parts=[line])
            w.record('line')
        w.save('acqueduct')

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
            return [array[p_array[i]:p_array[i+1]]
                    for i in range(len(p_array)-1)]

        # given a cell returns the edges implicitely defined in it
        def pairwise(seq):
            return [seq[i:i+2] for i in range(len(seq)-2)] + \
                [[seq[0], seq[len(seq)-1]]]

        # add edges to the graph
        for cell in chuncker(reader.elem2node, reader.p_elem2node):
            for u, v in pairwise(cell):
                if u not in self.graph[v]:
                    self.graph.add_edge(u, v, weight=self.distance3D(u, v))

    def distance(self, nodei, nodej):
        xi = nodei[0]
        yi = nodei[1]
        xj = nodej[0]
        yj = nodej[1]
        if len(nodei) == 3 and len(nodej) == 3:
            zi = nodei[2]
            zj = nodej[2]
            return math.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) +
                             (zi-zj)*(zi-zj))
        return math.sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))

    # cartesian norme in 2D
    def distance2D(self, nodei, nodej):
        xi = self.graph.nodes(data=True)[nodei][1]['pos'][0]
        yi = self.graph.nodes(data=True)[nodei][1]['pos'][1]
        xj = self.graph.nodes(data=True)[nodej][1]['pos'][0]
        yj = self.graph.nodes(data=True)[nodej][1]['pos'][1]
        return math.sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))

    # cartesian norme in 3D
    def distance3D(self, nodei, nodej):
        xi = self.graph.nodes(data=True)[nodei][1]['pos'][0]
        yi = self.graph.nodes(data=True)[nodei][1]['pos'][1]
        zi = self.graph.nodes(data=True)[nodei][1]['pos'][2]
        xj = self.graph.nodes(data=True)[nodej][1]['pos'][0]
        yj = self.graph.nodes(data=True)[nodej][1]['pos'][1]
        zj = self.graph.nodes(data=True)[nodej][1]['pos'][2]
        return math.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj))

    # returns 2D coordinates of the nodes of self.graph
    def coord2D(self, G):
        coord2D = {}
        for key, value in nx.get_node_attributes(G,
                                                 'pos').iteritems():
            coord2D[key] = [value[0], value[1]]
        return coord2D

    # display the mesh using networkx function
    def display_mesh(self):
        nodelist = []
        node_color = []
        for node in self.graph.nodes(data=1):
            node_type = node[1]['FID']
            if not node_type == '':
                nodelist.append(node[0])
                node_color.append('r' if node_type == 'sink' else 'b')
        try:
            nx.draw_networkx(self.graph, pos=self.coord2D(), nodelist=nodelist,
                             with_labels=0, node_color=node_color)
        except:
            pass

    # display the mesh with a path marked on it
    def display_path(self, path):
        nodelist = []
        node_color = []
        for node in self.graph.nodes(data=1):
            node_type = node[1]['FID']
            if not node_type == '':
                nodelist.append(node[0])
                node_color.append('r' if node_type == 'sink' else 'b')
                
        color = {edge: 'b' for edge in self.graph.edges()}
        # returns an array of pairs, the elements of seq two by two
        def pairwise(seq):
            return [seq[i:i+2] for i in range(len(seq)-2)]
        # colors the edges
        for u, v in pairwise(path):
            if (u, v) in color:
                color[(u, v)] = 'r'
            if (v, u) in color:
                color[(v, u)] = 'r'

        # makes an array of the dictionary, the order is importatant!
        array = []
        for edge in self.graph.edges():
            array += color[edge]
        nx.draw_networkx(self.graph, pos=self.coord2D(),
                               nodelist=nodelist, with_labels=0, 
                               node_color=node_color, edge_color=array)

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
        path_edges = [path[i:i+2] for i in range(len(path)-2)]
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
                    G.add_edge(n1, n2, {'path': [n1, n2], 'dist': self.distance(n1,n2)})

    def gabriel_graph(self, G):
        # condition to create the gabriel relative neighbour graph
        def neighbors(p, q, G):
            for r in G.nodes:
                if max(distance2D(p,r), distance2D(q,r)) < distance2D(p,q):
                    return False
            return True
        # connect graph
        for n1 in G.nodes():
            for n2 in G.nodes():
                if neighbors(n1, n2, G):
                    G.add_edge(n1, n2)

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

    def clusters(self, G):
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
        print ms.labels_

        # --- ADDUCTION ---
        '''
        # add cluster centers to the graph
        for node in cluster_centers:
            attribute = {'label': 'water tower', 'pos': node}
            G.add_node(node, attribute)
            attribute = {'type' : 'sink'}
            self.sinksource_graph.add_node(node, attribute)
        '''
        adduction = nx.Graph()
        cluster_centers = [(node[0], node[1]) for node in ms.cluster_centers_]
        for node in cluster_centers:
            adduction.add_node(node)
        self.complete_graph(adduction)
        adduction = nx.minimum_spanning_tree(adduction, weight='dist')
        # coord = {elem[0]: [elem[0][0], elem[0][1]] for elem in adduction.nodes(data=True)}
        # nx.draw_networkx(adduction, pos=coord, label=False)
        self.acqueduct.add_edges_from(adduction.edges())

        # --- DISTRIBUTION ---
        # add label info to the graph
        nx.set_node_attributes(G, 'label', labels)
        # initialize distribution graphs
        distribution = [nx.Graph() for cluster in cluster_centers]
        for node in labels:
            cluster = labels[node]
            distribution[cluster].add_node(node)
        '''
        # connect each node with his the cluster center
        node_list = []
        for index, node in enumerate(G):
            node_list.append(node)
            labels = nx.get_node_attributes(G, 'label')
            label = labels[node]
            if label is not 'water tower':
                G.add_edge(node, cluster_centers[label])
        '''
        for dist_graph in distribution:
            self.complete_graph(dist_graph)
            dist_graph = nx.minimum_spanning_tree(dist_graph, weight='dist')
            self.acqueduct.add_edges_from(dist_graph.edges())

        # coord = {elem[0]: [elem[0][0], elem[0][1]] for elem in self.acqueduct.nodes(data=True)}
        # nx.draw_networkx_edges(self.acqueduct, pos=coord)
        
# National__Hydrography__Dataset_NHD_Points_Medium_Resolution/National__Hydrography__Dataset_NHD_Points_Medium_Resolution
# National__Hydrography__Dataset_NHD_Lines_Medium_Resolution/National__Hydrography__Dataset_NHD_Lines_Medium_Resolution
# Railroads/Railroads
# Routesnaples/routesnaples
# "shapefiles/Domain", "shapefiles/pointspoly"
# "shapeline/shapeline", "shapeline/points"
router = Router(building_file="paesi/paesi")
router.clusters(router.graph)

# nx.draw_networkx(router.sinksource_graph, pos=router.coord2D(), with_labels=0)
# router.design_minimal_aqueduct()
# router.display_path(path)
# nx.draw_networkx_nodes(router.graph, pos=router.coord2D())
# router.display_mesh()
# print nx.clustering(router.graph)
# print nx.floyd_warshall_numpy(router.graph, weight='dist')
# runfile('/Users/Conrad/Documents/EC/Course deuxièmme année/Project Inno/Projet_P5C006/router.py', wdir='/Users/Conrad/Documents/EC/Course deuxièmme année/Project Inno/Projet_P5C006')