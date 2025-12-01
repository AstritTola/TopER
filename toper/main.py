import gc
import time
import platform
import numpy as np
import networkx as nx
import joblib
import pandas as pd
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

from .TopER_feature_extraction import *
from .threshold_extraction import *
from .threshold_reduction import *

def main(van_graph_list, funct_list, num_segm=501):
    # van_graph_list:
    #   A list of NetworkX graphs. For each graph, the code extracts raw threshold
    #   values (degree, centrality, curvature, weight, etc.) and builds filtered
    #   versions of the graph needed for feature computation.
    
    # funct_list:
    #   A list of strings indicating which filtration functions should be executed.
    #   Only the functions whose names appear in this list will be computed.
    #   Example entries: "degree", "deg_cen", "popularity", "closeness",
    #   "forman ricci", "ollivier ricci", "weight".
    
    # num_segm:
    #   Number of segments used to compress/reduce the raw threshold list into a
    #   fixed number of evenly spaced thresholds (default: 501). This controls the
    #   resolution of the filtration.
    
    # Additional behavior:
    # - Ollivier–Ricci ("olricci") is skipped automatically on Windows systems,
    #   because the required library does not run on Windows.
    #
    # - For each selected feature, the function measures runtime and prints it to
    #   the console for monitoring performance.
    #
    # - compute_and_save_features returns two lists: sub-level and super-level
    #   filtrations. These are stored in the final dictionary using keys in the
    #   format "<feature>_sub" and "<feature>_super".
    #
    # - The function returns a dictionary collecting all computed results for every
    #   requested feature.


    results = {}
    is_windows = platform.system().lower() == "windows"
    num_graph = len(van_graph_list)

    # Helper to measure runtime
    def time_it(label, func):
        start = time.time()
        out = func()
        end = time.time()
        print(f"[Runtime] {label}: {end - start:.3f} sec")
        return out

    # ---- Degree Centrality ----
    if "deg_cen" in funct_list:
        def run_degcen():
            graph_list, thresholds = get_thresholds_deg_cen(van_graph_list)
            thresholds = reduce_thresholds(thresholds, num_segm)
            F_sub, F_super = compute_and_save_features(
                "deg_cen", num_graph, graph_list, thresholds,
                sub_node_feat, super_node_feat, "degree_centrality"
            )
            return {"deg_cen_sub": F_sub, "deg_cen_super": F_super}

        results.update(time_it("Degree Centrality", run_degcen))
        gc.collect()

    # ---- Popularity ----
    if "popularity" in funct_list:
        def run_popularity():
            graph_list, thresholds = get_thresholds_popularity(van_graph_list)
            thresholds = reduce_thresholds(thresholds, num_segm)
            F_sub, F_super = compute_and_save_features(
                "popularity", num_graph, graph_list, thresholds,
                sub_node_feat, super_node_feat, "popularity"
            )
            return {"popularity_sub": F_sub, "popularity_super": F_super}

        results.update(time_it("Popularity", run_popularity))
        gc.collect()

    # ---- Closeness ----
    if "closeness" in funct_list:
        def run_closeness():
            graph_list, thresholds = get_thresholds_closeness(van_graph_list)
            thresholds = reduce_thresholds(thresholds, num_segm)
            F_sub, F_super = compute_and_save_features(
                 "closeness", num_graph, graph_list, thresholds,
                sub_node_feat, super_node_feat, "closeness"
            )
            return {"closeness_sub": F_sub, "closeness_super": F_super}

        results.update(time_it("Closeness", run_closeness))
        gc.collect()

    # ---- Forman–Ricci ----
    if "fricci" in funct_list:
        def run_forricci():
            graph_list, thresholds = get_thresholds_forricci(van_graph_list)
            thresholds = reduce_thresholds(thresholds, num_segm)
            F_sub, F_super = compute_and_save_features(
                "forricci", num_graph, graph_list, thresholds,
                sub_edge_feat, super_edge_feat, "formanCurvature"
            )
            return {"forricci_sub": F_sub, "forricci_super": F_super}

        results.update(time_it("Forman–Ricci", run_forricci))
        gc.collect()

    # ---- Ollivier–Ricci ----
    if "oricci" in funct_list:
        if is_windows:
            print("[Skip] Ollivier–Ricci is not supported on Windows.")
        else:
            def run_olricci():
                graph_list, thresholds = get_thresholds_olricci(van_graph_list)
                thresholds = reduce_thresholds(thresholds, num_segm)
                F_sub, F_super = compute_and_save_features(
                     "olricci", num_graph, graph_list, thresholds,
                    sub_edge_feat, super_edge_feat, "ricciCurvature"
                )
                return {"olricci_sub": F_sub, "olricci_super": F_super}

            results.update(time_it("Ollivier–Ricci", run_olricci))
            gc.collect()

    # ---- Degree ----
    if "degree" in funct_list:
        def run_degree():
            graph_list, thresholds = get_thresholds_degree(van_graph_list)
            thresholds = reduce_thresholds(thresholds, num_segm)
            F_sub, F_super = compute_and_save_features(
                 "degree", num_graph, graph_list, thresholds,
                sub_node_feat, super_node_feat, "degree"
            )
            return {"degree_sub": F_sub, "degree_super": F_super}

        results.update(time_it("Degree", run_degree))
        gc.collect()

    # ---- Weight ----
    if "weight" in funct_list:
        def run_weight():
            graph_list, thresholds = get_thresholds_weight(van_graph_list)
            thresholds = reduce_thresholds(thresholds, num_segm)
            F_sub, F_super = compute_and_save_features(
                 "weight", num_graph, graph_list, thresholds,
                sub_node_feat, super_node_feat, "weight"
            )
            return {"weight_sub": F_sub, "weight_super": F_super}

        results.update(time_it("Weight", run_weight))
        gc.collect()

    return results
