#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 22:57:30 2017

@author: Conrad
"""
import networkx as nx
from mesh import *
import math

class Router ( object ):
    
    # ----------------------------------------------------------------------------
    # -- CLASS ATTRIBUTES
    # ----------------------------------------------------------------------------

    # Error status
    SUCCESS = 0
    FAILURE = 1

    # Class description
    CLASS_NAME = "Router"
    CLASS_AUTHOR = "Marcello Vaccarino"
    METHODS = """
        __init__ ( self )

        ReadFromFileVtk (
            self,
            file_name )
        """
    
    # Attributes
    graph = nx.Graph()
    
    # ----------------------------------------------------------------------------
    # -- INITIALIZATION
    # ----------------------------------------------------------------------------

    def __init__ ( self, file_name ) :
        
        #initialize the vtk reader
        reader = Mesh()
        
        #read the vtk
        reader.ReadFromFileVtk(file_name) 
        
        #add nodes to the graph
        for index, node in enumerate(reader.node_coord):
            self.graph.add_node(index, pos = reader.node_coord[index])
        
        '''
        chuncker
        Principe basé sur le stockage CSR ou CRS (Compressed Row Storage) dont voici une illustration :
        Soient six nœuds numérotes de 0 à 5 et quatre  ́el ́ements form ́es par les nœuds (0, 2, 3) pour l’ ́el ́ement 0, (1, 2, 4) pour l’ ́el ́ement 1, (0, 1, 3, 4) pour l’ ́el ́e- ment 2 et (1, 5) pour l’ ́el ́ement 3. 
        Deux tableaux sont utilis ́es, l’un pour stocker de fa ̧con contigu ̈e les listes de nœuds qui composent les  ́el ́ements (table 1), l’autre pour indiquer la position, dans ce tableau, ou` commence chacune de ces listes (table 2). 
        Ainsi, le chiffre 6 en position 2 dans le tableau p elem2node indique que le premier nœud de l’ ́el ́ement 2 se trouve en position 6 du tableau elem2node. La derni`ere valeur dans p elem2node correspond au nombre de cellules (la taille) du tableau elem2node.
        
        elem2node
        0 | 2 | 3 | 1 | 2 | 4 | 0 | 1 | 3 | 4 | 1 | 5  
        1   2   3   4   5   6   7   8   9   10  11  12    
        ^       ^           ^               ^       ^
        
        p_elem2node
        0 | 3 | 6 | 10 | 12   
        1   2   3    4    4                                                  '''
        
        def chuncker(array, p_array):
            return [array[p_array[i]:p_array[i+1]] for i in range(len(p_array)-1)]
        
        #given a cell returns the edges implicitely defined in it
        def pairwise(seq):
            return [seq[i:i+2] for i in range(len(seq)-2)] + [[seq[0], seq[len(seq)-1]]]
        
        #add edges to the graph  
        for cell in chuncker(reader.elem2node, reader.p_elem2node):
            for u, v in pairwise(cell):
                if not u in self.graph[v] :
                    self.graph.add_edge(u, v, weight = self.distance3D(u, v))
        
    def display(self):
        try: 
            path = nx.shortest_path(graph, source=0, target=2, weight = "weight") 
        except:
            print("no path")                    
        else:
            print path
            color = {}
            for edge in graph.edges():
                color[edge] = 'b'
            for i in range(0, len(path)-1):
                edge1 = (path[i], path[i+1])
                if edge1 in color.keys():
                    color[edge1] = 'r'
            
            print path
            array = []
            for edge in graph.edges():
                array += color[edge]
            #nx.draw_networkx_edges(graph , pos = coord2D, edge_color =  array)
            nx.draw_networkx(graph , pos = coord2D, edge_color = array)
        
    #cartesian norme in 2D
    def distance2D(self, nodei, nodej):
        xi = graph.nodes(data=True)[nodei][1]['pos'][0]
        yi = graph.nodes(data=True)[nodei][1]['pos'][1]
        xj = graph.nodes(data=True)[nodej][1]['pos'][0]
        yj = graph.nodes(data=True)[nodej][1]['pos'][1]
        return math.sqrt((xi-xj)*(xi-xj)+(yi-yj)*(yi-yj))
    
    #cartesian norme in 3D
    def distance3D(self, nodei, nodej):
        xi = graph.nodes(data=True)[nodei][1]['pos'][0]
        yi = graph.nodes(data=True)[nodei][1]['pos'][1]
        zi = graph.nodes(data=True)[nodei][1]['pos'][2]
        xj = graph.nodes(data=True)[nodej][1]['pos'][0]
        yj = graph.nodes(data=True)[nodej][1]['pos'][1]
        zj = graph.nodes(data=True)[nodei][1]['pos'][2]
        return math.sqrt((xi-xj)*(xi-xj) + (yi-yj)*(yi-yj) + (zi-zj)*(zi-zj))
    
    def coord2D(self):
        coord2D = nx.get_node_attributes(self.graph, 'pos')
        for key in coord2D:
            coord2D[key] = [coord2D[key][0], coord2D[key][1]]
        return coord2D
            
    def display_mesh(self):
        nx.draw_networkx_edges(graph , pos = self.coord2D())
        
    def display_path(self, path):
        color = {edge: 'b' for edge in graph.edges()}

        def pairwise(seq):
            return [seq[i:i+2] for i in range(len(seq)-2)]

        for u, v in pairwise(path):
            print u, v
            if color.has_key((u,v)):
                color[(u,v)] = 'r'
            if color.has_key((v,u)):
                color[(v,u)] = 'r'

        array = []
        for edge in graph.edges():
            array += color[edge]
        nx.draw_networkx_edges(graph , pos = coord2D, edge_color =  array)
    
    def shortest_path(self):
        try: 
            path = nx.shortest_path(graph, source=0, target=2, weight = "weight") 
        except:
            print("no path")
        return path
    
    def _main_ ():
        router = Router("meshvtk.vtk")
        path = router.shortest_path()
        router.display_path(path)