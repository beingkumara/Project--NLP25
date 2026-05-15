import json
import os
import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_PATH = "cranfield/"
OUT_DIR = "output/"


def load_json_file(path):
    # First we check the base condition for safety
    # Let us implement the logic to load the json file
    f = open(path, "r")
    data = json.load(f)
    f.close()
    return data


def save_json_file(data, path):
    # First we check the base condition for safety
    # Basically executing the logic to save the json file
    f = open(path, "w")
    json.dump(data, f, indent=4)
    f.close()
    print("  Saved:", path)


def get_query_length_groups(all_queries_list):
    # Here we calculate the main metric for length
    # Executing logic for getting query length groups
    length_info = {}
    for i in range(len(all_queries_list)):
        q_entry = all_queries_list[i]
        q_id = int(q_entry["query number"])
        q_text = q_entry["query"]
        words = q_text.strip().split()
        num_words = len(words)
        if num_words <= 5:
            group_name = "short"
        elif num_words <= 10:
            group_name = "medium"
        else:
            group_name = "long"
        length_info[q_id] = {
            "length_words": num_words,
            "length_group": group_name,
            "query_text": q_text,
        }
    return length_info


def get_qrel_density_groups(all_qrels_list):
    # Executing logic for getting qrel density groups
    relevant_count_per_query = {}
    for i in range(len(all_qrels_list)):
        entry = all_qrels_list[i]
        q_id = int(entry["query_num"])
        pos = entry["position"]
        if pos == 1 or pos == 2 or pos == 3 or (pos == 4):
            if q_id not in relevant_count_per_query:
                relevant_count_per_query[q_id] = 0
            relevant_count_per_query[q_id] = relevant_count_per_query[q_id] + 1
    density_info = {}
    for q_id in relevant_count_per_query:
        count = relevant_count_per_query[q_id]
        if count <= 3:
            group_name = "sparse"
        elif count <= 9:
            group_name = "medium"
        else:
            group_name = "dense"
        density_info[q_id] = {"relevant_count": count, "density_group": group_name}
    return density_info


def compute_class_ap(per_query_ap_data, query_ids_list, class_info_dict, class_key):
    # Basically executing logic to compute class ap
    all_class_values = []
    for q_id in class_info_dict:
        class_val = class_info_dict[q_id][class_key]
        if class_val not in all_class_values:
            all_class_values.append(class_val)
    system_names = []
    for key in per_query_ap_data.keys():
        if key != "query_ids":
            system_names.append(key)
    result = {}
    for sys_idx in range(len(system_names)):
        sys_name = system_names[sys_idx]
        ap_list = per_query_ap_data[sys_name]
        class_ap_sums = {}
        class_counts = {}
        for cv in all_class_values:
            class_ap_sums[cv] = 0.0
            class_counts[cv] = 0
        for q_idx in range(len(query_ids_list)):
            q_id = query_ids_list[q_idx]
            ap_val = ap_list[q_idx]
            if q_id in class_info_dict:
                class_val = class_info_dict[q_id][class_key]
                class_ap_sums[class_val] = class_ap_sums[class_val] + ap_val
                class_counts[class_val] = class_counts[class_val] + 1
        sys_result = {}
        for cv in all_class_values:
            if class_counts[cv] > 0:
                mean_ap = class_ap_sums[cv] / float(class_counts[cv])
            else:
                mean_ap = 0.0
            sys_result[cv] = {"count": class_counts[cv], "mean_ap": round(mean_ap, 4)}
        result[sys_name] = sys_result
    return result


def add_baseline_to_class_ap(
    class_ap_result, per_query_ap_data, query_ids_list, class_info_dict, class_key
):

    # Let us add baseline to class ap here
    baseline_key = "VSM (Baseline)"
    if baseline_key not in per_query_ap_data:
        print("  Warning: VSM (Baseline) not found in per_query_ap_data")
        return class_ap_result
    if baseline_key in class_ap_result:
        print("  Baseline already included in result.")
    else:
        print(
            "  Note: VSM (Baseline) was not in class result, it may already be included."
        )
    return class_ap_result


def plot_class_bar_chart(class_ap_result, class_order, chart_title, x_label, out_path):
    # Executing logic to plot the class bar chart
    system_names = list(class_ap_result.keys())
    num_systems = len(system_names)
    num_classes = len(class_order)
    bar_width = 0.8 / float(num_systems)
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = []
    for i in range(num_classes):
        x_positions.append(float(i))
    for sys_idx in range(num_systems):
        sys_name = system_names[sys_idx]
        sys_data = class_ap_result[sys_name]
        map_values = []
        for cv in class_order:
            if cv in sys_data:
                map_values.append(sys_data[cv]["mean_ap"])
            else:
                map_values.append(0.0)
        bar_positions = []
        for i in range(len(x_positions)):
            offset = (sys_idx - num_systems / 2.0 + 0.5) * bar_width
            bar_positions.append(x_positions[i] + offset)
        ax.bar(bar_positions, map_values, bar_width, label=sys_name)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_order)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean AP@10")
    ax.set_title(chart_title)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("  Saved chart:", out_path)


def find_universal_failures(
    per_query_ap_data, query_ids_list, length_info, density_info, all_queries_list
):
    # Finding universal failures logic goes here
    system_names = []
    for key in per_query_ap_data.keys():
        if key != "query_ids":
            system_names.append(key)
    raw_text_dict = {}
    for q_entry in all_queries_list:
        raw_text_dict[int(q_entry["query number"])] = q_entry["query"]
    universal_failures = []
    for q_idx in range(len(query_ids_list)):
        q_id = query_ids_list[q_idx]
        all_zero = True
        for sys_idx in range(len(system_names)):
            sys_name = system_names[sys_idx]
            ap_val = per_query_ap_data[sys_name][q_idx]
            if ap_val > 0.001:
                all_zero = False
                break
        if all_zero == True:
            entry = {"query_id": q_id, "query_text": raw_text_dict.get(q_id, "N/A")}
            if q_id in length_info:
                entry["length_words"] = length_info[q_id]["length_words"]
                entry["length_group"] = length_info[q_id]["length_group"]
            else:
                entry["length_words"] = -1
                entry["length_group"] = "unknown"
            if q_id in density_info:
                entry["relevant_count"] = density_info[q_id]["relevant_count"]
                entry["density_group"] = density_info[q_id]["density_group"]
            else:
                entry["relevant_count"] = 0
                entry["density_group"] = "no_qrels"
            universal_failures.append(entry)
    return universal_failures


def compute_marginal_gap_breakdown(
    per_query_ap_data,
    query_ids_list,
    length_info,
    density_info,
    system_a_name,
    system_b_name,
):
    # Basically calculating the marginal gap breakdown
    if system_a_name not in per_query_ap_data:
        print("  Warning:", system_a_name, "not in per_query_ap_data")
        return {}
    if system_b_name not in per_query_ap_data:
        print("  Warning:", system_b_name, "not in per_query_ap_data")
        return {}
    ap_a = per_query_ap_data[system_a_name]
    ap_b = per_query_ap_data[system_b_name]
    length_groups = ["short", "medium", "long"]
    density_groups = ["sparse", "medium", "dense"]
    length_breakdown = {}
    for lg in length_groups:
        length_breakdown[lg] = {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0}
    density_breakdown = {}
    for dg in density_groups:
        density_breakdown[dg] = {"a_wins": 0, "b_wins": 0, "ties": 0, "count": 0}
    for q_idx in range(len(query_ids_list)):
        q_id = query_ids_list[q_idx]
        ap_val_a = ap_a[q_idx]
        ap_val_b = ap_b[q_idx]
        if ap_val_a > ap_val_b + 0.001:
            winner = "a"
        elif ap_val_b > ap_val_a + 0.001:
            winner = "b"
        else:
            winner = "tie"
        if q_id in length_info:
            lg = length_info[q_id]["length_group"]
            if lg in length_breakdown:
                length_breakdown[lg]["count"] = length_breakdown[lg]["count"] + 1
                if winner == "a":
                    length_breakdown[lg]["a_wins"] = length_breakdown[lg]["a_wins"] + 1
                elif winner == "b":
                    length_breakdown[lg]["b_wins"] = length_breakdown[lg]["b_wins"] + 1
                else:
                    length_breakdown[lg]["ties"] = length_breakdown[lg]["ties"] + 1
        if q_id in density_info:
            dg = density_info[q_id]["density_group"]
            if dg in density_breakdown:
                density_breakdown[dg]["count"] = density_breakdown[dg]["count"] + 1
                if winner == "a":
                    density_breakdown[dg]["a_wins"] = (
                        density_breakdown[dg]["a_wins"] + 1
                    )
                elif winner == "b":
                    density_breakdown[dg]["b_wins"] = (
                        density_breakdown[dg]["b_wins"] + 1
                    )
                else:
                    density_breakdown[dg]["ties"] = density_breakdown[dg]["ties"] + 1
    for lg in length_breakdown:
        total = length_breakdown[lg]["count"]
        if total > 0:
            a_pct = round(100.0 * length_breakdown[lg]["a_wins"] / float(total), 1)
            b_pct = round(100.0 * length_breakdown[lg]["b_wins"] / float(total), 1)
            length_breakdown[lg]["a_win_pct"] = a_pct
            length_breakdown[lg]["b_win_pct"] = b_pct
        else:
            length_breakdown[lg]["a_win_pct"] = 0.0
            length_breakdown[lg]["b_win_pct"] = 0.0
    for dg in density_breakdown:
        total = density_breakdown[dg]["count"]
        if total > 0:
            a_pct = round(100.0 * density_breakdown[dg]["a_wins"] / float(total), 1)
            b_pct = round(100.0 * density_breakdown[dg]["b_wins"] / float(total), 1)
            density_breakdown[dg]["a_win_pct"] = a_pct
            density_breakdown[dg]["b_win_pct"] = b_pct
        else:
            density_breakdown[dg]["a_win_pct"] = 0.0
            density_breakdown[dg]["b_win_pct"] = 0.0
    result = {
        "system_a": system_a_name,
        "system_b": system_b_name,
        "by_query_length": length_breakdown,
        "by_qrel_density": density_breakdown,
    }
    return result


def enrich_failure_analysis(failure_report, length_info, density_info):
    # Executing logic to enrich failure analysis
    enriched = {}
    system_names = list(failure_report.keys())
    for sys_idx in range(len(system_names)):
        sys_name = system_names[sys_idx]
        sys_data = failure_report[sys_name]
        new_failures_list = sys_data["new_failures"]
        length_fail_counts = {"short": 0, "medium": 0, "long": 0}
        density_fail_counts = {"sparse": 0, "medium": 0, "dense": 0}
        enriched_new_failures = []
        for fail_idx in range(len(new_failures_list)):
            fail_entry = new_failures_list[fail_idx]
            q_id = fail_entry["query_id"]
            enriched_entry = {}
            for k in fail_entry:
                enriched_entry[k] = fail_entry[k]
            if q_id in length_info:
                enriched_entry["length_group"] = length_info[q_id]["length_group"]
                enriched_entry["length_words"] = length_info[q_id]["length_words"]
                lg = length_info[q_id]["length_group"]
                if lg in length_fail_counts:
                    length_fail_counts[lg] = length_fail_counts[lg] + 1
            else:
                enriched_entry["length_group"] = "unknown"
                enriched_entry["length_words"] = -1
            if q_id in density_info:
                enriched_entry["density_group"] = density_info[q_id]["density_group"]
                enriched_entry["relevant_count"] = density_info[q_id]["relevant_count"]
                dg = density_info[q_id]["density_group"]
                if dg in density_fail_counts:
                    density_fail_counts[dg] = density_fail_counts[dg] + 1
            else:
                enriched_entry["density_group"] = "unknown"
                enriched_entry["relevant_count"] = -1
            enriched_new_failures.append(enriched_entry)
        enriched[sys_name] = {
            "new_failures_count": sys_data["new_failures_count"],
            "fixed_failures_count": sys_data["fixed_failures_count"],
            "failures_by_length_group": length_fail_counts,
            "failures_by_density_group": density_fail_counts,
            "new_failures": enriched_new_failures,
            "fixed_failures": sys_data["fixed_failures"],
        }
    return enriched


def main():
    # Let us make sure we don't divide by zero
    # Basically the main function to run the logic
    print("=" * 60)
    print("Query Class Analysis Script")
    print("=" * 60)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    print("\n[1/7] Loading input data...")
    full_pq_ap_path = OUT_DIR + "part5_full_per_query_ap.json"
    if not os.path.exists(full_pq_ap_path):
        print("  ERROR: File not found:", full_pq_ap_path)
        print("  Please run part5.py first to generate this file.")
        return
    per_query_ap_data = load_json_file(full_pq_ap_path)
    query_ids_list = per_query_ap_data["query_ids"]
    print("  Loaded per-query AP for", len(query_ids_list), "queries")
    print("  Systems found:", [k for k in per_query_ap_data.keys() if k != "query_ids"])
    all_queries = load_json_file(DATASET_PATH + "cran_queries.json")
    print("  Loaded", len(all_queries), "raw queries")
    all_qrels = load_json_file(DATASET_PATH + "cran_qrels.json")
    print("  Loaded", len(all_qrels), "qrel entries")
    failure_report = load_json_file(OUT_DIR + "part5_failure_analysis.json")
    print("  Loaded existing failure analysis")
    print("\n[2/7] Computing query length groups...")
    length_info = get_query_length_groups(all_queries)
    short_count = 0
    medium_count = 0
    long_count = 0
    for q_id in length_info:
        grp = length_info[q_id]["length_group"]
        if grp == "short":
            short_count = short_count + 1
        elif grp == "medium":
            medium_count = medium_count + 1
        else:
            long_count = long_count + 1
    print("  Short queries (<=5 words):", short_count)
    print("  Medium queries (6-10 words):", medium_count)
    print("  Long queries (>=11 words):", long_count)
    print("\n[3/7] Computing qrel density groups...")
    density_info = get_qrel_density_groups(all_qrels)
    sparse_count = 0
    medium_den_count = 0
    dense_count = 0
    no_qrel_count = 0
    for q_id in query_ids_list:
        if q_id in density_info:
            dg = density_info[q_id]["density_group"]
            if dg == "sparse":
                sparse_count = sparse_count + 1
            elif dg == "medium":
                medium_den_count = medium_den_count + 1
            else:
                dense_count = dense_count + 1
        else:
            no_qrel_count = no_qrel_count + 1
    print("  Sparse (1-3 relevant docs):", sparse_count)
    print("  Medium (4-9 relevant docs):", medium_den_count)
    print("  Dense (>=10 relevant docs):", dense_count)
    print("  No qrel entry:", no_qrel_count)
    print("\n[4/7] Computing per-class AP analysis...")
    print("  Computing length-class AP breakdown...")
    length_class_ap = compute_class_ap(
        per_query_ap_data, query_ids_list, length_info, "length_group"
    )
    print("  Computing density-class AP breakdown...")
    density_class_ap = compute_class_ap(
        per_query_ap_data, query_ids_list, density_info, "density_group"
    )
    print("\n  --- Length Class Analysis (MAP@10 by query length) ---")
    sys_names_for_print = list(length_class_ap.keys())
    print(
        "  {:25s} {:>10s} {:>10s} {:>10s}".format("System", "Short", "Medium", "Long")
    )
    for s in sys_names_for_print:
        sh = length_class_ap[s].get("short", {}).get("mean_ap", 0.0)
        me = length_class_ap[s].get("medium", {}).get("mean_ap", 0.0)
        lo = length_class_ap[s].get("long", {}).get("mean_ap", 0.0)
        print("  {:25s} {:>10.4f} {:>10.4f} {:>10.4f}".format(s[:25], sh, me, lo))
    print("\n  --- Density Class Analysis (MAP@10 by qrel density) ---")
    print(
        "  {:25s} {:>10s} {:>10s} {:>10s}".format("System", "Sparse", "Medium", "Dense")
    )
    for s in sys_names_for_print:
        sp = density_class_ap[s].get("sparse", {}).get("mean_ap", 0.0)
        me = density_class_ap[s].get("medium", {}).get("mean_ap", 0.0)
        de = density_class_ap[s].get("dense", {}).get("mean_ap", 0.0)
        print("  {:25s} {:>10.4f} {:>10.4f} {:>10.4f}".format(s[:25], sp, me, de))
    print("\n[5/7] Generating class analysis charts...")
    plot_class_bar_chart(
        length_class_ap,
        class_order=["short", "medium", "long"],
        chart_title="Mean AP@10 by Query Length Group — All Systems",
        x_label="Query Length Group",
        out_path=OUT_DIR + "part5_length_class_chart.png",
    )
    plot_class_bar_chart(
        density_class_ap,
        class_order=["sparse", "medium", "dense"],
        chart_title="Mean AP@10 by Qrel Density Group — All Systems",
        x_label="Qrel Density Group (Number of Relevant Documents)",
        out_path=OUT_DIR + "part5_density_class_chart.png",
    )
    print("\n[5b/7] Finding universal failures (AP=0 for all systems)...")
    universal_failures = find_universal_failures(
        per_query_ap_data, query_ids_list, length_info, density_info, all_queries
    )
    print(
        "  Found", len(universal_failures), "queries where ALL systems score AP@10 = 0"
    )
    for uf in universal_failures:
        print(
            "    Query",
            uf["query_id"],
            "(" + uf["length_group"] + ",",
            uf.get("density_group", "?") + "):",
            uf["query_text"][:70],
        )
    print("\n[6/7] Computing marginal gap breakdowns...")
    prf_vs_bm25 = compute_marginal_gap_breakdown(
        per_query_ap_data,
        query_ids_list,
        length_info,
        density_info,
        "BM25 + PRF",
        "BM25",
    )
    print("  PRF vs BM25 breakdown computed.")
    print("  By length:")
    for lg in prf_vs_bm25["by_query_length"]:
        entry = prf_vs_bm25["by_query_length"][lg]
        print(
            "    " + lg + ": PRF wins",
            entry["a_wins"],
            "(" + str(entry["a_win_pct"]) + "%)",
            "BM25 wins",
            entry["b_wins"],
            "(" + str(entry["b_win_pct"]) + "%)",
        )
    print("  By density:")
    for dg in prf_vs_bm25["by_qrel_density"]:
        entry = prf_vs_bm25["by_qrel_density"][dg]
        print(
            "    " + dg + ": PRF wins",
            entry["a_wins"],
            "(" + str(entry["a_win_pct"]) + "%)",
            "BM25 wins",
            entry["b_wins"],
            "(" + str(entry["b_win_pct"]) + "%)",
        )
    bm25_vs_lsa = compute_marginal_gap_breakdown(
        per_query_ap_data,
        query_ids_list,
        length_info,
        density_info,
        "BM25",
        "LSA (k=200, CV)",
    )
    print("\n  BM25 vs LSA breakdown computed.")
    print("  By length:")
    for lg in bm25_vs_lsa["by_query_length"]:
        entry = bm25_vs_lsa["by_query_length"][lg]
        print(
            "    " + lg + ": BM25 wins",
            entry["a_wins"],
            "(" + str(entry["a_win_pct"]) + "%)",
            "LSA wins",
            entry["b_wins"],
            "(" + str(entry["b_win_pct"]) + "%)",
        )
    print("\n[7/7] Enriching failure analysis with boundary conditions...")
    enriched_failures = enrich_failure_analysis(
        failure_report, length_info, density_info
    )
    print("  Boundary condition summary:")
    for sys_name in enriched_failures:
        fr = enriched_failures[sys_name]
        if fr["new_failures_count"] > 0:
            print(
                "  " + sys_name + " new failures by length:",
                fr["failures_by_length_group"],
            )
            print(
                "  " + sys_name + " new failures by density:",
                fr["failures_by_density_group"],
            )
    print("\n[Saving] Writing all outputs to disk...")
    save_json_file(length_class_ap, OUT_DIR + "part5_query_length_class_analysis.json")
    save_json_file(
        density_class_ap, OUT_DIR + "part5_query_density_class_analysis.json"
    )
    save_json_file(universal_failures, OUT_DIR + "part5_universal_failures.json")
    save_json_file(prf_vs_bm25, OUT_DIR + "part5_prf_vs_bm25_breakdown.json")
    save_json_file(bm25_vs_lsa, OUT_DIR + "part5_lsa_vs_bm25_breakdown.json")
    save_json_file(
        enriched_failures, OUT_DIR + "part5_failure_boundary_conditions.json"
    )
    save_json_file(length_info, OUT_DIR + "part5_query_length_info.json")
    save_json_file(density_info, OUT_DIR + "part5_query_density_info.json")
    print("\n" + "=" * 60)
    print("DONE. All analysis saved to output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
