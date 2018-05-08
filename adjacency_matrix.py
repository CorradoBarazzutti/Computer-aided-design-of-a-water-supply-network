import networkx as nx
import matplotlib as plt

def read_adjacency(filename):
	with open(filename) as fin:
		lines = fin.readlines()
	lines = [line.rstrip('\n') for line in lines]
	lines = [line.split(" ") for line in lines]
	print(lines)
	G = nx.Graph()
	for n1, line in enumerate(lines):
		for n2, edge in enumerate(line):
			G.add_edge(n1, n2, weight=edge)
	nx.draw_networkx(G)

file_path = "/Users/conrad/Documents/EC/Course_deuxiemme_annee/Project_Inno/Projet_P5C006/geographycal_data/Howgrp.txt"
read_adjacency(file_path)