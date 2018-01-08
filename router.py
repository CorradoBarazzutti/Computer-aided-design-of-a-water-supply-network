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

    # ----------------------------------------------------------------------------
    # -- INITIALIZATION
    # ----------------------------------------------------------------------------

    def __init__(self, file_name):
        if not isinstance(file_name, str):
            raise ValueError("file_name must be of type String")
        self.read_shp(file_name)

    # ----------------------------------------------------------------------------
    # -- CLASS ATTRIBUTES
    # ----------------------------------------------------------------------------
    def read_shp(self, file_name):
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
                    print '{0}: {1}, {2}'.format(i, n1, n2)
                    self.graph.add_edge(n1, n2, attribute)

            if shape.shapeType == 5:    # polygraph
                # cuncker: see commpresed row storage
                def chuncker(array, p_array):
                    p_array.append(len(array))
                    return [array[p_array[i]:p_array[i+1]]
                            for i in range(len(p_array)-1)]

                # given a cell returns the edges (node touple) implicitely defined in it
                def pairwise(seq):
                    return [seq[i:i+2] for i in range(len(seq)-1)]

                for cell in chuncker(shape.points, shape.parts):
                    print cell
                    print pairwise(cell)
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
    def coord2D(self):
        coord2D = {}
        for key, value in nx.get_node_attributes(self.graph,
                                                 'pos').iteritems():
            coord2D[key] = [value[0], value[1]]
        return coord2D

    # display the mesh using networkx function
    def display_mesh(self):
        nx.draw_networkx_edges(self.graph, pos=self.coord2D())

    # display the mesh with a path marked on it
    def display_path(self, path):
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
        nx.draw_networkx_edges(self.graph, pos=self.coord2D(),
                               edge_color=array)

    def shortest_path(self, node1, node2):
        """
        Calculates the shortest path on self.graph.
        Path is a sequence of traversed nodes
        """
        try:
            path = nx.shortest_path(self.graph, source=node1, target=node2,
                                    weight="weight")
        except:
            print("no path")
        return path

    def path_lenght(self, path):
        """
        given a path on the graph returns the lenght of the path in the
        unit the coordinats are expressed
        """
        lenght = 0.0
        # given the path (list of node) returns the edges contained
        path_edges = [path[i:i+2] for i in range(len(path)-2)]
        # itereate to edges and calculate the weight
        for u, v in path_edges:
            lenght += self.distance3D(u, v)
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

    def compute_source_matrix(self):
        # from scipy.sparse import csgraph
        # self.adjacency_matrix = csgraph.csgraph_from_dense([[1 if jnode in environment.getNeighbours(inode) else 0 for jnode in environment.graph] for inode in environment.graph])
        self.dist_matrix, self.predecessors = csgraph.floyd_warshall(self.adjacency_matrix, directed = False, return_predecessors = True, unweighted = True)
        
        self.cheese_dist_matrix = []
        
        if not environment.graph[environment.start_node]['cheese']:
            L = [0]
            for i, inode in enumerate(environment.graph):
                if inode == environment.start_node:
                    break
            j=0
            for jnode in environment.graph:
                if environment.graph[jnode]['cheese']:
                    L.append(self.dist_matrix[i][j])
                j+=1
            self.cheese_dist_matrix.append(L)
            vocabulary = {0: i}
        
        k=0
        for i, inode in enumerate(environment.graph):
            if environment.graph[inode]['cheese'] :
                j=0
                L = []
                for jnode in environment.graph:
                    if environment.graph[jnode]['cheese'] or jnode == environment.start_node:
                        L.append(self.dist_matrix[i][j])
                    j+=1 
                self.cheese_dist_matrix.append(L)  
                vocabulary.update({k+1: i})
                k+=1
                    
# National__Hydrography__Dataset_NHD_Points_Medium_Resolution/National__Hydrography__Dataset_NHD_Points_Medium_Resolution
# National__Hydrography__Dataset_NHD_Lines_Medium_Resolution/National__Hydrography__Dataset_NHD_Lines_Medium_Resolution
# Railroads/Railroads
# Routesnaples/routesnaples
# shapefiles/Domain
# shapeline/shapeline
router = Router("shapefiles/Domain")
# router.display_path(path)
# nx.draw_networkx_nodes(router.graph, pos=router.coord2D())
router.display_mesh()
# print nx.clustering(router.graph)
# print nx.floyd_warshall_numpy(router.graph, weight='dist')
