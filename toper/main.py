def main(van_graph_list):
    num_graph = len(van_graph_list)

    print("1 Degree Centrality")

    graph_list_deg_cen, thresholds_deg_cen = get_thresholds_deg_cen(van_graph_list)
    thresholds_deg_cen = reduce_thresholds(thresholds_deg_cen)

    F_degcen = compute_and_save_features(
        name,
        "deg_cen",
        num_graph,
        graph_list_deg_cen,
        thresholds_deg_cen,
        sub_node_feat,
        super_node_feat,
        "degree_centrality",
    )

    return F_degcen
