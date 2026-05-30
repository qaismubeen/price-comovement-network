
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
warnings.filterwarnings("ignore")

# bold helper so section headings print bold in terminal without any symbols
BOLD  = "\033[1m"
RESET = "\033[0m"

def load_data(filepath="cpi_data.csv"):
    """Load CSV and reshape into long format: City | Item | Category | Year | Month | Price"""
    # read the csv file into a dataframe
    df = pd.read_csv(filepath)

    # Strip column name spaces; rename leading "4City" to "City"
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: "City"}, inplace=True)

    # Melt month columns into rows
    # build list of month column names Month1 to Month12
    month_cols = [f"Month{i}" for i in range(1, 13)]
    # melt so each month becomes a separate row instead of a column
    df_long = df.melt(
        id_vars=["City", "Item", "Category", "Year"],
        value_vars=month_cols,
        var_name="MonthLabel",
        value_name="Price"
    )

    # We will be Extracting the  month number
    # strip the word "Month" and keep only the number, convert to int
    df_long["Month"] = df_long["MonthLabel"].str.replace("Month", "").astype(int)
    df_long.drop(columns=["MonthLabel"], inplace=True)
    # sort rows so data is ordered by city, item, year, then month
    df_long.sort_values(["City", "Item", "Year", "Month"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # Now to Fill missing prices with forward fill within group
    # fill gaps in price using previous value, then backward fill any remaining
    df_long["Price"] = (
        df_long.groupby(["City", "Item", "Year"])["Price"]
        .transform(lambda x: x.ffill().bfill())
    )

    # print summary of what was loaded so user can verify
    print(f"Data loaded: {df_long.shape[0]} rows")
    print(f"Cities   : {sorted(df_long['City'].unique())}")
    print(f"Items    : {sorted(df_long['Item'].unique())}")
    print(f"Years    : {sorted(df_long['Year'].unique())}")
    print(f"Categories: {sorted(df_long['Category'].unique())}")
    return df_long

# chv = compute_change_vectors
def chv(df_long):
    """
    For each (City, Item, Year), compute the 11-element monthly change vector:
    delta_p_m = price_m - price_(m-1)  for months 2 through 12
    Returns dict: {(city, item, year): np.array of shape (11,)}
    """
    # empty dict to store change vectors keyed by (city, item, year)
    vectors = {}
    # group data by city, item, year to process each combination separately
    grouped = df_long.groupby(["City", "Item", "Year"])

    for (city, item, year), grp in grouped:
        # sort by month to make sure prices are in order before differencing
        grp_sorted = grp.sort_values("Month")
        prices = grp_sorted["Price"].values
        if len(prices) == 12:
            # compute month-to-month differences, gives 11 values
            delta = np.diff(prices)
            vectors[(city, item, year)] = delta
        else:
            # Pad or skip if data incomplete
            if len(prices) > 1:
                delta = np.diff(prices)
                # create zero array of length 11 and fill with available diffs
                padded = np.zeros(11)
                padded[:len(delta)] = delta
                vectors[(city, item, year)] = padded

    print(f"\nChange vectors computed: {len(vectors)} (city, item, year) combinations")
    return vectors


# sim_city = compute_similarity_per_city
def sim_city(vectors, year):
    """
    For a given year, compute cosine similarity between every pair of items in every city.
    Returns dict: {(city, item_i, item_j): similarity_value}
    """
    # we will get all unique cities and items for this year
    # filter only keys that belong to this year
    year_keys = [(c, i, y) for (c, i, y) in vectors if y == year]
    cities = sorted(set(c for c, i, y in year_keys))
    items  = sorted(set(i for c, i, y in year_keys))

    # dict to store similarity score for each (city, item_i, item_j) triple
    sim_res = {}

    for city in cities:
        # Build matrix: rows = items, cols = change vector
        item_vecs = {}
        for item in items:
            key = (city, item, year)
            # only add item if its vector exists for this city and year
            if key in vectors:
                item_vecs[item] = vectors[key]

        avail_items = list(item_vecs.keys())
        # need at least 2 items to form any pair
        if len(avail_items) < 2:
            continue

        # iterate over all unique pairs of items in this city
        for item_i, item_j in combinations(avail_items, 2):
            v_i = item_vecs[item_i].reshape(1, -1)
            v_j = item_vecs[item_j].reshape(1, -1)
            # compute norms to check for zero vectors before cosine calc
            norm_i = np.linalg.norm(v_i)
            norm_j = np.linalg.norm(v_j)

            if norm_i == 0 or norm_j == 0:
                # if either vector is all zeros similarity is set to 0
                sim = 0.0
            else:
                sim = float(cosine_similarity(v_i, v_j)[0][0])

            # store result keyed by (city, item_i, item_j)
            sim_res[(city, item_i, item_j)] = sim

    return sim_res, cities, items


# cnt_sim = count_similar_cities
def cnt_sim(sim_res, cities, items, tau=0.7):
    """
    For each item pair, count how many cities have cosine similarity >= tau.
    Also compute average similarity across all cities.
    Returns:
        N_dict: {(item_i, item_j): count of cities with sim >= tau}
        avg_dict: {(item_i, item_j): average similarity across all cities}
    """
    N_dict   = {}
    avg_dict = {}

    # go through every unique pair of items
    for item_i, item_j in combinations(items, 2):
        count = 0
        sims  = []
        # check each city to see if this pair crosses the threshold
        for city in cities:
            key = (city, item_i, item_j)
            if key in sim_res:
                s = sim_res[key]
                sims.append(s)
                # count city only if similarity meets or exceeds tau
                if s >= tau:
                    count += 1

        # store count and average for this item pair
        N_dict[(item_i, item_j)]   = count
        avg_dict[(item_i, item_j)] = np.mean(sims) if sims else 0.0

    return N_dict, avg_dict


def build_graph(items, N_dict, avg_dict, df_long, K=2, weight_method="count"):
    """
    Build undirected graph where edge exists if N(i,j) >= K.
    weight_method: 'count' => w = N(i,j)  |  'avg' => w = average similarity
    """
    # create empty undirected graph
    G = nx.Graph()

    # Adding  all the  items as nodes with category attribute
    # build a lookup dict from item name to its category
    item_cat = (
        df_long[["Item", "Category"]].drop_duplicates()
        .set_index("Item")["Category"].to_dict()
    )
    # add each item as a node, attach category as node attribute
    for item in items:
        G.add_node(item, category=item_cat.get(item, "Unknown"))

    # Add the edges
    for (item_i, item_j), count in N_dict.items():
        # only add edge if enough cities have high similarity
        if count >= K:
            # choose weight based on method: city count or avg similarity
            if weight_method == "count":
                w = count
            else:
                w = avg_dict.get((item_i, item_j), 0.0)
            # store both weight, city_count, and avg_sim as edge attributes
            G.add_edge(item_i, item_j, weight=w, city_count=count,
                       avg_sim=avg_dict.get((item_i, item_j), 0.0))

    return G


# cent = compute_centrality
def cent(G):
    """Compute degree, closeness, and betweenness centrality."""
    # return empty dicts if graph has no nodes at all
    if G.number_of_nodes() == 0:
        return {}, {}, {}

    # degree centrality measures how connected each node is relative to all others
    deg_c     = nx.degree_centrality(G)
    # closeness measures how quickly a node can reach all others
    close_c   = nx.closeness_centrality(G)
    # betweenness measures how often a node appears on shortest paths between others
    between_c = nx.betweenness_centrality(G, normalized=True)

    return deg_c, close_c, between_c


# print_cent = print_centrality_report
def print_cent(G, year, deg_c, close_c, between_c, top_n=5):
    """Print top items by each centrality measure."""
    # print bold section header for this year's graph analysis
    print(f"\n{BOLD}Graph Analysis - Year {year}{RESET}")
    print(f"Nodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")
    # density only makes sense if there are edges
    if G.number_of_edges() > 0:
        print(f"Density: {nx.density(G):.4f}")

    if not deg_c:
        print("  (Graph is empty or disconnected)")
        return

    # helper to return top n items sorted by value descending
    def top(d, n=top_n):
        return sorted(d.items(), key=lambda x: -x[1])[:n]

    # print top items for each centrality type
    print(f"\n  Top {top_n} by Degree Centrality:")
    for item, val in top(deg_c):
        print(f"    {item:<25} {val:.4f}")

    print(f"\n  Top {top_n} by Closeness Centrality:")
    for item, val in top(close_c):
        print(f"    {item:<25} {val:.4f}")

    print(f"\n  Top {top_n} by Betweenness Centrality:")
    for item, val in top(between_c):
        print(f"    {item:<25} {val:.4f}")


CATEGORY_COLORS = {
    "Food"      : "#4e79a7",
    "Energy"    : "#f28e2b",
    "Clothing"  : "#59a14f",
    "Housing"   : "#e15759",
    "Education" : "#76b7b2",
    "Health"    : "#edc948",
    "Transport" : "#b07aa1",
    "Unknown"   : "#bab0ac",
}


def draw_graph(G, year, tau, K, weight_method, save_path=None):
    """Draw the graph with category-based node coloring."""
    # skip drawing if no nodes exist in this graph
    if G.number_of_nodes() == 0:
        print(f"  [Year {year}] Graph is empty, skipping visualization.")
        return

    plt.figure(figsize=(14, 10))
    # set the main title showing year and parameter settings
    plt.title(
        f"Item Price Co-Movement Network  |  Year {year}\n"
        f"tau={tau}, K={K}, weight={weight_method}",
        fontsize=14, pad=15
    )

    # compute spring layout positions, seed keeps layout consistent across runs
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # The color of notes by category
    # map each node to a color based on its category attribute
    node_colors = [
        CATEGORY_COLORS.get(G.nodes[n].get("category", "Unknown"), "#bab0ac")
        for n in G.nodes()
    ]

    # Edge widths scaled by weight
    # normalize edge weights so widths range from 1 to 4
    weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [1 + 3 * (w / max_w) for w in weights]

    # draw nodes, labels, and edges separately for more control
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color="#555555")

    # build legend patches from category colors, exclude Unknown
    leg_patches = [
        mpatches.Patch(color=color, label=cat)
        for cat, color in CATEGORY_COLORS.items()
        if cat != "Unknown"
    ]
    plt.legend(handles=leg_patches, loc="upper left", fontsize=8,
               title="Category", title_fontsize=9)
    plt.axis("off")
    plt.tight_layout()

    # save to disk if a path was provided, overwrite if file already exists
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# temp_ana = temporal_analysis
def temp_ana(graphs, years):
    """Compare graphs across years."""
    # print bold section heading for temporal analysis
    print(f"\n{BOLD}Temporal Analysis{RESET}")

    # convert each year's edge list to a set of frozensets for easy set operations
    edge_sets = {}
    for y, G in zip(years, graphs):
        edge_sets[y] = set(frozenset(e) for e in G.edges())

    # Stable edges
    if len(years) == 3:
        # intersect all three years to find edges that always exist
        stable = edge_sets[years[0]] & edge_sets[years[1]] & edge_sets[years[2]]
        print(f"\n  Edges stable across ALL 3 years: {len(stable)}")
        for e in sorted(stable, key=lambda x: sorted(x)):
            items_in_edge = sorted(list(e))
            print(f"    {items_in_edge[0]}  <-->  {items_in_edge[1]}")

    # New edges each year
    # compare consecutive years to find gained and lost edges
    for i in range(1, len(years)):
        new_edges  = edge_sets[years[i]] - edge_sets[years[i-1]]
        lost_edges = edge_sets[years[i-1]] - edge_sets[years[i]]
        print(f"\n  Year {years[i-1]} -> Year {years[i]}:")
        print(f"    New edges: {len(new_edges)}")
        print(f"    Lost edges: {len(lost_edges)}")

    # Degree centrality stability
    # find which item had highest degree each year to track dominance over time
    print(f"\n  Top item by Degree Centrality per year:")
    for y, G in zip(years, graphs):
        if G.number_of_edges() > 0:
            deg = nx.degree_centrality(G)
            top_item = max(deg, key=deg.get)
            print(f"    Year {y}: {top_item} ({deg[top_item]:.4f})")


# cat_ana = category_analysis
def cat_ana(G, year):
    """Analyze within-category vs between-category edges."""
    # print bold section heading for this year's category analysis
    print(f"\n{BOLD}Category Analysis - Year {year}{RESET}")

    within  = 0
    between = 0

    # loop over all edges and check if both endpoints share the same category
    for u, v in G.edges():
        cat_u = G.nodes[u].get("category", "Unknown")
        cat_v = G.nodes[v].get("category", "Unknown")
        if cat_u == cat_v:
            within += 1
        else:
            between += 1

    # print both counts and their percentage of total edges
    total = within + between
    print(f"  Within-category edges : {within}  ({100*within/total:.1f}% of total)" if total else "  No edges.")
    print(f"  Between-category edges: {between}  ({100*between/total:.1f}% of total)" if total else "")


# sens_ana = sensitivity_analysis
def sens_ana(vectors, df_long, year, cities, items):
    """Vary tau and K, observe edge count changes."""
    # print bold section heading for sensitivity analysis
    print(f"\n{BOLD}Sensitivity Analysis - Year {year}{RESET}")

    # compute similarity results for the given year
    sim_res, _, _ = sim_city(vectors, year)

    # iterate over combinations of tau and K to see how edge count changes
    print(f"\n  {'tau':<8} {'K':<6} {'Edges':<8} {'Nodes with edges'}")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        N_dict, avg_dict = cnt_sim(sim_res, cities, items, tau=tau)
        for K in [1, 2, 3]:
            G_test = build_graph(items, N_dict, avg_dict, df_long, K=K)
            # count nodes that have at least one edge
            nodes_with_edges = sum(1 for n in G_test.nodes() if G_test.degree(n) > 0)
            print(f"  {tau:<8} {K:<6} {G_test.number_of_edges():<8} {nodes_with_edges}")


# plot_cent_cmp = plot_centrality_comparison
def plot_cent_cmp(graphs, years, centrality_type="degree", save_path=None):
    """Bar chart comparing centrality of items across 3 years."""
    # collect all unique items across all year graphs
    all_items = set()
    for G in graphs:
        all_items.update(G.nodes())
    all_items = sorted(all_items)

    # one subplot per year, all in one row
    fig, axes = plt.subplots(1, len(years), figsize=(18, 6), sharey=False)
    fig.suptitle(f"{centrality_type.capitalize()} Centrality Comparison Across Years", fontsize=14)

    for ax, G, year in zip(axes, graphs, years):
        # compute the requested centrality type for this year's graph
        if centrality_type == "degree":
            c = nx.degree_centrality(G)
        elif centrality_type == "closeness":
            c = nx.closeness_centrality(G)
        else:
            c = nx.betweenness_centrality(G)

        # sort items by centrality and take top 10
        items_sorted = sorted(c.keys(), key=lambda x: -c[x])[:10]
        vals = [c[i] for i in items_sorted]

        # horizontal bar chart, reversed so highest value is at top
        ax.barh(items_sorted[::-1], vals[::-1], color="#4e79a7", alpha=0.85)
        ax.set_title(f"Year {year}", fontsize=11)
        ax.set_xlabel("Centrality Value")
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    # save and overwrite if file exists at that path
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# plot_tau_cmp = plot_threshold_comparison
def plot_tau_cmp(vectors, df_long, year, cities, items, save_path=None):
    """Show graph with two different thresholds side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Threshold Comparison - Year {year}", fontsize=14)

    # compute similarities once and reuse for both tau values
    sim_res, _, _ = sim_city(vectors, year)

    for ax, (tau, K) in zip(axes, [(0.6, 2), (0.85, 3)]):
        # build graph for this particular tau and K combination
        N_dict, avg_dict = cnt_sim(sim_res, cities, items, tau=tau)
        G = build_graph(items, N_dict, avg_dict, df_long, K=K)

        pos = nx.spring_layout(G, seed=42, k=2.5)
        # assign category colors to each node
        node_colors = [
            CATEGORY_COLORS.get(G.nodes[n].get("category", "Unknown"), "#bab0ac")
            for n in G.nodes()
        ]
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=500,
                with_labels=True, font_size=6, font_weight="bold",
                edge_color="#555555", width=1.5, alpha=0.85)
        ax.set_title(f"tau={tau}, K={K}  |  Edges: {G.number_of_edges()}", fontsize=11)

    plt.tight_layout()
    # save and overwrite if file already exists
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# plot_wt_cmp = plot_weight_comparison
def plot_wt_cmp(vectors, df_long, year, cities, items, tau=0.7, K=2, save_path=None):
    """Side-by-side: count-weighted vs avg-similarity-weighted graph."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Weighting Scheme Comparison - Year {year}", fontsize=14)

    # compute similarities and counts once, reuse for both weighting methods
    sim_res, _, _ = sim_city(vectors, year)
    N_dict, avg_dict = cnt_sim(sim_res, cities, items, tau=tau)

    for ax, method in zip(axes, ["count", "avg"]):
        # build a separate graph for each weighting method
        G = build_graph(items, N_dict, avg_dict, df_long, K=K, weight_method=method)
        pos = nx.spring_layout(G, seed=42, k=2.5)

        # color nodes by their category
        node_colors = [
            CATEGORY_COLORS.get(G.nodes[n].get("category", "Unknown"), "#bab0ac")
            for n in G.nodes()
        ]
        # scale edge widths proportionally to their weight values
        weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [0.5 + 3.5 * (w / max_w) for w in weights]

        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=500,
                with_labels=True, font_size=6, font_weight="bold",
                edge_color="#555555", width=edge_widths, alpha=0.85)
        label = "City Count Weight" if method == "count" else "Avg Similarity Weight"
        ax.set_title(f"{label}  |  Edges: {G.number_of_edges()}", fontsize=11)

    plt.tight_layout()
    # save and overwrite if file already exists
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# plot_3yr = plot_3year_comparison
def plot_3yr(graphs, years, save_path=None):
    """Side-by-side graphs for all 3 years."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle("Graph Evolution Across 3 Years", fontsize=14)

    # draw one graph per subplot, one per year
    for ax, G, year in zip(axes, graphs, years):
        pos = nx.spring_layout(G, seed=42, k=2.5)
        # color each node by its category
        node_colors = [
            CATEGORY_COLORS.get(G.nodes[n].get("category", "Unknown"), "#bab0ac")
            for n in G.nodes()
        ]
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=450,
                with_labels=True, font_size=5.5, font_weight="bold",
                edge_color="#555555", width=1.2, alpha=0.85)
        ax.set_title(f"Year {year}  |  Edges: {G.number_of_edges()}", fontsize=11)

    # Legend
    # build one legend for all subplots at the bottom of the figure
    leg_patches = [
        mpatches.Patch(color=color, label=cat)
        for cat, color in CATEGORY_COLORS.items()
        if cat != "Unknown"
    ]
    fig.legend(handles=leg_patches, loc="lower center", ncol=7,
               fontsize=8, title="Category")
    # leave space at the bottom for the legend
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    # save and overwrite if file already exists
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
    plt.close()


# The main body
def main():
    # Parameters
    TAU = 0.7    # cosine similarity threshold
    K   = 2      # minimum number of cities
    WEIGHT_METHOD = "count"   # "count" or "avg"

    # print bold title at start of program
    print(f"{BOLD}Item-Level Price Co-Movement Network using CPI{RESET}")

    # Loading the  data
    # get the folder where this .py file lives so all paths are relative to it
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "cpi_data.csv")

    # load and reshape the csv data into long format
    df_long = load_data(file_path)

    # chv = compute_change_vectors
    # compute month-to-month price change vectors for every city-item-year combo
    vectors = chv(df_long)

    # get sorted list of all unique years present in the data
    years = sorted(df_long["Year"].unique())
    print(f"\nYears in dataset: {years}")

    # lists to accumulate graphs and centralities for each year
    yr_graphs = []
    yr_cents  = []

    for year in years:
        # print bold year processing header
        print(f"\n{BOLD}Processing Year {year}  (tau={TAU}, K={K}, weight={WEIGHT_METHOD}){RESET}")

        # Similarity per city
        # compute pairwise cosine similarity for all item pairs in each city
        sim_res, cities, items = sim_city(vectors, year)

        # Count cities where sim >= tau
        # count how many cities cross the threshold for each item pair
        N_dict, avg_dict = cnt_sim(sim_res, cities, items, tau=TAU)

        # Building the graph
        # edges are added only where enough cities share similar price movement
        G = build_graph(items, N_dict, avg_dict, df_long, K=K, weight_method=WEIGHT_METHOD)
        yr_graphs.append(G)

        # Centrality
        # compute three centrality measures for all nodes in this year's graph
        deg_c, close_c, between_c = cent(G)
        yr_cents.append((deg_c, close_c, between_c))
        print_cent(G, year, deg_c, close_c, between_c)

        # Category analysis
        # check how many edges are within vs across categories
        cat_ana(G, year)

        # Visualize this year's graph
        # save PNG to the same folder as this script, overwrite if exists
        draw_graph(G, year, TAU, K, WEIGHT_METHOD,
                   save_path=os.path.join(BASE_DIR, f"graph_year_{year}.png"))

    #Temporal analysis
    # compare edge sets and centrality across all years
    temp_ana(yr_graphs, years)

    #Sensitivity analysis using first year
    # recompute similarity for first year to use in sensitivity test
    sim_res_0, cities_0, items_0 = sim_city(vectors, years[0])
    sens_ana(vectors, df_long, years[0], cities_0, items_0)

    # Multi-year comparison plots
    print(f"\n{BOLD}Generating comparison plots...{RESET}")
    # save 3-year side by side graph to script folder, overwrite if exists
    plot_3yr(yr_graphs, years,
             save_path=os.path.join(BASE_DIR, "comparison_3years.png"))

    # Threshold comparison (year 1)
    # recompute similarity for first year for threshold comparison plot
    sim_res_y1, cities_y1, items_y1 = sim_city(vectors, years[0])
    plot_tau_cmp(vectors, df_long, years[0], cities_y1, items_y1,
                 save_path=os.path.join(BASE_DIR, "comparison_thresholds.png"))

    # Weight comparison (year 1)
    # compare count-weighted vs avg-weighted graph side by side
    plot_wt_cmp(vectors, df_long, years[0], cities_y1, items_y1,
                tau=TAU, K=K,
                save_path=os.path.join(BASE_DIR, "comparison_weights.png"))

    # Centrality comparison
    # plot degree and betweenness centrality bar charts across all years
    plot_cent_cmp(yr_graphs, years, "degree",
                  save_path=os.path.join(BASE_DIR, "centrality_degree.png"))
    plot_cent_cmp(yr_graphs, years, "betweenness",
                  save_path=os.path.join(BASE_DIR, "centrality_betweenness.png"))

    print(f"\n{BOLD}All done! Check the saved .png files.{RESET}")


if __name__ == "__main__":
    main()
