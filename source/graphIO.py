import math
import matplotlib.pyplot as plt
import networkx as nx

class graph_reader():
    """
    This class provides the functions for graph reading from a multitude of formats
    """
    def __init__(self, graph):
        self.graph = graph

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
                # pack attributes
                attributes = dict(zip(fields[1:], shapeRecs.record))
                center_coord = self.avg(cell)
                attributes['pos'] = center_coord
                # add nodes
                self.graph.add_node(center_coord)
                for key, value in attributes.items():
                    self.graph.nodes[center_coord][key] = value

    # chuncker: see commpresed row storage
    def row_chuncker(self, array, p_array):
        p_array.append(len(array))
        return [array[p_array[i]:p_array[i + 1]]
                for i in range(len(p_array) - 1)]

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
            if shape.shapeType == 1:  # point
                attributes = dict(zip(fields, shapeRecs.record))
                attributes["pos"] = shape.points[0]
                self.graph.add_node(shape.points[0], attributes)

            if shape.shapeType == 3:  # polylines
                attributes1 = dict(zip(fields, shapeRecs.record))
                attributes2 = dict(zip(fields, shapeRecs.record))
                for i in range(len(shape.points) - 1):
                    '''
                    attributes1["pos"] = shape.points[i]
                    n1 = self.add_node_unique(shape.points[i], attributes1)
                    attributes2["pos"] = shape.points[i + 1]
                    n2 = self.add_node_unique(shape.points[i + 1], attributes2)
                    attribute = {'dist': self.distance(n1, n2)}
                    self.graph.add_edge(n1, n2, attribute) '''
                    attributes1["pos"] = shape.points[i]
                    n1 = shape.points[i]
                    self.graph.add_node(n1, attributes1)
                    attributes2["pos"] = shape.points[i + 1]
                    n2 = shape.points[i + 1]
                    self.graph.add_node(n2, attributes2)
                    attribute = {'dist': self.distance(n1, n2)}
                    self.graph.add_edge(n1, n2, attribute)

            if shape.shapeType == 5:  # polygraph
                # chuncker: see commpresed row storage
                def chuncker(array, p_array):
                    p_array.append(len(array))
                    return [array[p_array[i]:p_array[i + 1]]
                            for i in range(len(p_array) - 1)]

                # given a cell returns the edges (node touple) implicitely defined in it
                def pairwise(seq):
                    return [seq[i:i + 2] for i in range(len(seq) - 1)]

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
                if shape.shapeType != 1:  # point
                    raise ValueError("point_file must be of type 1: points")
                new_attributes = dict(zip(new_fields, shapeRecs.record))
                nodes = [tuple(point) for point in shape.points]
                for node in nodes:
                    nodes2attributes[node].update(new_attributes)
            for node, data in self.graph.nodes(data=1):
                data.update(nodes2attributes[node])
            # nx.set_node_attributes(self.graph, nodes2attributes)

    def read_adjacency(self, filename):
        with open(filename) as fin:
            lines = fin.readlines()
        lines = [line.rstrip('\n') for line in lines]
        lines = [line.split(" ") for line in lines]

        adjacency_metrix = []
        coordinates = []
        for i, line in enumerate(lines):
            if line == ['']:
                pass
            if line == ["[ADJACENCY_MATRIX]"]:
                nodes_number = len(lines[i + 1])
                for adj_line in lines[i + 1:i + nodes_number + 1]:
                    adjacency_metrix.append(adj_line)
            if line == ["[COORDINATES]"]:
                j = 1
                while len(lines[i + j]) == 2:
                    coordinates.append(lines[i + j])
                    j += 1

        for i, line in enumerate(adjacency_metrix):
            for j, edge in enumerate(line):
                if float(edge) > 0:
                    node1 = (float(coordinates[i][0]), float(coordinates[i][1]))
                    node2 = (float(coordinates[j][0]), float(coordinates[j][1]))
                    self.graph.add_edge(node1, node2, weight=float(edge))

    def read_epanet(self, filename):
        with open(filename) as fin:
            lines = fin.readlines()
        lines = [line.rstrip('\n') for line in lines]
        lines = [line.split(" ") for line in lines]

        adjacency_metrix = []
        coordinates = []
        for i, line in enumerate(lines):
            if line == ['']:
                pass
            if line == ["[ADJACENCY_MATRIX]"]:
                nodes_number = len(lines[i + 1])
                for adj_line in lines[i + 1:i + nodes_number + 1]:
                    adjacency_metrix.append(adj_line)
            if line == ["[COORDINATES]"]:
                j = 1
                while len(lines[i + j]) == 2:
                    coordinates.append(lines[i + j])
                    j += 1

        for i, line in enumerate(adjacency_metrix):
            for j, edge in enumerate(line):
                if float(edge) > 0:
                    node1 = (float(coordinates[i][0]), float(coordinates[i][1]))
                    node2 = (float(coordinates[j][0]), float(coordinates[j][1]))
                    self.graph.add_edge(node1, node2, weight=float(edge))

    def read_epanet(self, file_name):
        with open(file_name + "_nodes.txt", "r") as f_in:
            id2node = {id: node for id, (node, _) in enumerate(self.graph.nodes.items())}
            for line in f_in:
                line = line.split()
                if line[0] == "Junc" or line[0] == "Tank":
                    id = int(float(line[1]))
                    node = id2node[id]
                    pressure = line[4]
                    self.graph.nodes[node]["ELEVATION"] = float(line[2])
                    self.graph.nodes[node]["H"] = float(line[3])
                    self.graph.nodes[node]["PRESSURE"] = float(pressure)
        with open(file_name + "_edges.txt", "r") as f_in:
            id2edge = {id: edge for id, (edge, _) in enumerate(self.graph.edges.items())}
            for line in f_in:
                line = line.split()
                if line[0] == "Pipe":
                    id = int(float(line[1]))
                    edge = id2edge[id]
                    flow = float(line[4])
                    print(flow)
                    self.graph.edges[edge]["Q"] = float(flow)
                    self.graph.edges[edge]["DIAMETER"] = float(line[3])
                    self.graph.edges[edge]["V"] = float(line[5])

class graph_writer():
    pass


class display_graph():
    """
    This class implements some usefull display function to plot the acqueduct a graph
    """

    def __init__(self, graph):
        self.graph = graph

    def distance2D(self, nodei, nodej):
        """
        Computes the cartesian norme in 2D
        """
        xi = self.graph.nodes(data=True)[nodei][1]['pos'][0]
        yi = self.graph.nodes(data=True)[nodei][1]['pos'][1]
        xj = self.graph.nodes(data=True)[nodej][1]['pos'][0]
        yj = self.graph.nodes(data=True)[nodej][1]['pos'][1]
        return math.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj))

    def distance3D(self, nodei, nodej):
        """
        Computes the cartesian norme in 3D
        """
        xi = self.graph.nodes(data=True)[nodei][1]['pos'][0]
        yi = self.graph.nodes(data=True)[nodei][1]['pos'][1]
        zi = self.graph.nodes(data=True)[nodei][1]['pos'][2]
        xj = self.graph.nodes(data=True)[nodej][1]['pos'][0]
        yj = self.graph.nodes(data=True)[nodej][1]['pos'][1]
        zj = self.graph.nodes(data=True)[nodej][1]['pos'][2]
        return math.sqrt((xi - xj) * (xi - xj) + (yi - yj) * (yi - yj) + (zi - zj) * (zi - zj))

    def coord2D(self, G):
        """
        Returns 2D coordinates of the nodes of self.graph
        """
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
            plt.draw()
        except:
            pass

    def display_path(self, path):
        """
        Display the mesh with a path marked on it
        """
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
            return [seq[i:i + 2] for i in range(len(seq) - 2)]

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