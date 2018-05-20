import os

from router3 import Router
from kpi_calculator import kpi_calculator

from source import graphIO

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

last_slash = 0
for i, char in enumerate(PROJECT_PATH):
    if char == "/":
        last_slash = i
PROJECT_PATH = PROJECT_PATH[:last_slash]

def render_vtk(file_name):
    """
    Renders a vtk file
    """
    import vtk

    # Read the source file.
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()  # Needed because of GetScalarRange
    output = reader.GetOutput()
    scalar_range = output.GetScalarRange()

    # Create the mapper that corresponds the objects of the vtk.vtk file
    # into graphics elements
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputData(output)
    mapper.SetScalarRange(scalar_range)

    # Create the Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # Create the Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # Set background to white

    # Create the RendererWindow
    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)

    # Create the RendererWindowInteractor and display the vtk_file
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()

def vesuvio_example():
    """
    Example where the routing capabilities of the program are shown.
    A topograpy of the vesuvio area is imported from a vtk file and the shortest path no it is calculated.
    The result is than exported to vtk so that can be seen in paraview
    """
    router = Router(topo_file=PROJECT_PATH + "vtk/Vesuvio")
    router.routing(32729, 31991)
    # write to vtk
    router.write2vtk(router.acqueduct)
    # render_vtk("vtk/Vesuvio")

def paesi_example():
    """
    Example of water distribution network partitioning.
    The result is exported to a shapefile
    """
    router = Router(building_file=PROJECT_PATH + "geographycal_data/paesi_elev/paesi_elev")
    router.clusters(router.graph)
    router.write2shp(router.acqueduct, "acqueduct1")

def casdetude():
    """
    Imports the topograpy of the city of Monterusciello and automatically design the water supply network
    """
    file_path = PROJECT_PATH + "/geographycal_data/Monterusciello/MontEdo_buildings"
    router = Router(building_file=file_path)

    router.design_aqueduct(1)
    router.write2shp(router.acqueduct, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct")

    router.design_aqueduct(0)
    router.solve(router.acqueduct)
    router.write2epanet(router.acqueduct, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct")

    # read_epanet = graphIO.graph_reader(router.acqueduct)
    # read_epanet.read_epanet(PROJECT_PATH + "/geographycal_data/SolvedNet/distruptive")
    kpi_calculator(router.acqueduct)



def casdetude_dinardo():
    """
    Imports the topograpy of the city of Monterusciello and automatically design the water supply network
    """
    file_path = PROJECT_PATH + "/geographycal_data/Monterusciello/MontEdo_buildings"
    router = Router(building_file=file_path)

    router.design_aqueduct(0)

    router.solve(router.acqueduct)
    minimal = router.design_minimal_aqueduct(router.acqueduct, "Q*H")
    kpi_calculator(minimal)

    print("N   H   Z   P")
    for i, (node, datadict) in enumerate(router.acqueduct.nodes.items()):
        print(i, round(datadict["H"]), round(datadict["ELEVATION"]), round(datadict["H"] - datadict["ELEVATION"]))


    router.write2shp(minimal, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct")
    router.write2epanet(minimal, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct")


def casdetude_genetics():
    """
    Imports the topograpy of the city of Monterusciello and automatically design the water supply network
    """
    file_path = PROJECT_PATH + "/geographycal_data/Monterusciello/MontEdo_buildings"
    router = Router(building_file=file_path)

    router.design_aqueduct(0)

    router.write2epanet(router.acqueduct, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct",
                        diam=False)

    read_epanet = graphIO.graph_reader(router.acqueduct)
    read_epanet.read_epanet(PROJECT_PATH + "/geographycal_data/SolvedNet/MonteSolution")
    kpi_calculator(router.acqueduct)

    minimal = router.design_minimal_aqueduct(router.acqueduct, "Q*H")
    router.write2epanet(minimal, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct", diam=False)


def casdetude_genetics_sk():
    """
    Imports the topograpy of the city of Monterusciello and automatically design the water supply network
    """
    file_path = PROJECT_PATH + "/geographycal_data/Monterusciello/MontEdo_buildings"
    router = Router(building_file=file_path)

    router.design_aqueduct(0)

    router.write2epanet(router.acqueduct, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct",
                        diam=False)

    read_epanet = graphIO.graph_reader(router.acqueduct)
    read_epanet.read_epanet(PROJECT_PATH + "/geographycal_data/SolvedNet/MonteSolution")

    minimal = router.design_minimal_aqueduct(router.acqueduct, "Q*H")
    router.write2epanet(minimal, PROJECT_PATH + "/Monterusciello_solution/Monterusciello_acqueduct", diam=False)

    read_epanet = graphIO.graph_reader(router.acqueduct)
    read_epanet.read_epanet(PROJECT_PATH + "/geographycal_data/SolvedNet/prova")
    kpi_calculator(router.acqueduct)


def adjacency_matrix():
    """
    Imports a graph from an adjacency matrix and performs some spectra operations on it
    """
    file_path = PROJECT_PATH + "/geographycal_data/adjacency_matrix/Howgrp.txt"
    router = Router(adjacency_metrix=file_path)
    # router.write2vtk(router.graph, "adjacency_matrix")
    # nx.draw(router.graph)
    # plt.show()
    # adjacency matrix
    A = nx.adjacency_matrix(router.graph, weight=None).toarray()
    # ... and its spectrum
    nx.adjacency_spectrum(router.graph, weight=None)
    # weighted adjacency
    W = nx.adjacency_matrix(router.graph)
    # D
    I = np.reshape(np.ones(12), (-1, 1))
    D = np.matmul(A, I)
    # combinatorial graph Laplacian L = D - A
    L = nx.laplacian_matrix(router.graph, weight=None)
    # ... and his spectrum
    nx.laplacian_spectrum(router.graph, weight=None)
    # weighted Laplacian
    Y = nx.laplacian_matrix(router.graph)

    # Note
    sumD = np.matmul(I.transpose(), D)
    sumD = sumD[0][0]
    sumA = 0
    for row in np.nditer(A):
        for e in np.nditer(row):
            sumA += e

    # Fielder vector
    fiedler_vector = nx.fiedler_vector(router.graph, weight=None)

    # Matrix Double index Sum

    def D_app(F):
        return D * F

    def A_app(F):
        AF = np.zeros(len(F))
        for i, e_i in enumerate(F):
            for j, e_j in enumerate(F):
                if (A[i][j] != 0):
                    AF[i] += F[j]
        return AF

def automatic_partitioning():
    """
    Import a graph from an adjacency and runs the luvain comunity partitionig algorithm to find communites.
    The result is show at screen
    """
    def draw_labels(labels_vector):
        labs = {node: labels_vector[i] for i, node in enumerate(router.graph.nodes())}
        coord = {touple: list(touple) for touple in router.graph.nodes()}
        nx.draw_networkx(router.graph, coord, labels=labs)
        plt.show()

    file_path = PROJECT_PATH + "/geographycal_data/adjacency_matrix/Howgrp.txt"
    router = Router(adjacency_metrix=file_path)
    draw_labels(router.louvain_clustering(router.graph, weight='weight'))

# automatic_partitioning()
casdetude()
# net_solving_case()
