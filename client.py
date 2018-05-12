from router3 import Router

def render_vtk(file_name):
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


def tsp_example():
    return 0


def clustering_example():
    return 0


def template_clustering(path_sample, eps, minpts, amount_clusters=None, visualize=True, ccore=False):
    sample = read_sample(path_sample);

    optics_instance = optics(sample, eps, minpts, amount_clusters, ccore);
    (ticks, _) = timedcall(optics_instance.process);

    if (visualize is True):
        clusters = optics_instance.get_clusters();
        noise = optics_instance.get_noise();

        visualizer = cluster_visualizer();
        visualizer.append_clusters(clusters, sample);
        visualizer.append_cluster(noise, sample, marker='x');
        visualizer.show();

        ordering = optics_instance.get_ordering();
        analyser = ordering_analyser(ordering);

        ordering_visualizer.show_ordering_diagram(analyser, amount_clusters);


def vesuvio_example():
    router = Router(topo_file="vtk/Vesuvio")
    router.route_vesuvio(32729, 31991)
    # write to vtk
    router.write2vtk(router.acqueduct)
    # render_vtk("vtk/Vesuvio")


def paesi_example():
    router = Router(building_file="geographycal_data/paesi_elev/paesi_elev")
    router.clusters(router.graph)
    router.write2shp(router.acqueduct, "acqueduct1")


def cluster_simple_example():
    import random;

    from pyclustering.cluster import cluster_visualizer;

    from pyclustering.utils import read_sample, timedcall;

    from pyclustering.samples.definitions import SIMPLE_SAMPLES, FCPS_SAMPLES;

    template_clustering(SIMPLE_SAMPLES.SAMPLE_SIMPLE1, 0.5, 3);


def casdetude():
    file_path = "/Users/conrad/Documents/EC/Course_deuxiemme_annee/Project_Inno/Projet_P5C006/geographycal_data/Monterusciello/MontEdo_buildings"
    router = Router(building_file=file_path)

    router.design_aqueduct(0)

    router.solve(router.acqueduct)

    router.write2shp(router.acqueduct, "Monterusciello_acqueduct")
    router.write2epanet(router.acqueduct, "Monterusciello_acqueduct")


def adjacency_matrix():
    file_path = "/Users/conrad/Documents/EC/Course_deuxiemme_annee/Project_Inno/Projet_P5C006/geographycal_data/adjacency_matrix/Howgrp.txt"
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
    def draw_labels(labels_vector):
        labs = {node: labels_vector[i] for i, node in enumerate(router.graph.nodes())}
        coord = {touple: list(touple) for touple in router.graph.nodes()}
        nx.draw_networkx(router.graph, coord, labels=labs)
        plt.show()

    file_path = "/Users/conrad/Documents/EC/Course_deuxiemme_annee/Project_Inno/Projet_P5C006/geographycal_data" \
                "/adjacency_matrix/Howgrp.txt"
    router = Router(adjacency_metrix=file_path)
    draw_labels(router.louvain_clustering(router.graph, weight='weight'))

def bruna():
    file_path = "/Users/conrad/Documents/EC/Course_deuxiemme_annee/Project_Inno/Projet_P5C006/geographycal_data" \
                "/adjacency_matrix/Howgrp.txt"
    router = Router(adjacency_metrix=file_path)
    router.write2list("vert2vert")

# automatic_partitioning()
casdetude()
# net_solving_case()
