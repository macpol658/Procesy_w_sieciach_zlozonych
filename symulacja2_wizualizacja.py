"""
=============================================================================
ZINTEGROWANA SYMULACJA I WIZUALIZACJA SIECI ZŁOŻONEJ LOTNISK W POLSCE
=============================================================================
Źródło danych: ULC, "Szacowane przepustowości portów lotniczych", lipiec 2009
Narzędzia: SimPy, NetworkX, igraph, Matplotlib
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")  # Tryb bezokienkowy do bezpośredniego zapisu plików
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # Poprawiono brakujące 'as'
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import simpy
import networkx as nx
import igraph as ig
import random
import numpy as np
import collections
import itertools

# ── Deterministyczne ziarno dla powtarzalności ──────────────────────────────
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DANE WEJŚCIOWE
# =============================================================================
AIRPORTS = {
    "WAW": {"name": "Warszawa Okęcie",          "city": "Warszawa", "capacity_mpax": 9.5, "actual_mpax": 7.9, "gates": 12, "lat": 52.165, "lon": 20.967},
    "KRK": {"name": "Kraków J. Paweł II",        "city": "Kraków",   "capacity_mpax": 5.0, "actual_mpax": 2.8, "gates": 6,  "lat": 50.077, "lon": 19.784},
    "KTW": {"name": "Katowice-Pyrzowice",        "city": "Katowice", "capacity_mpax": 4.5, "actual_mpax": 2.6, "gates": 6,  "lat": 50.474, "lon": 19.080},
    "GDN": {"name": "Gdańsk im. Wałęsy",         "city": "Gdańsk",   "capacity_mpax": 3.5, "actual_mpax": 2.0, "gates": 5,  "lat": 54.377, "lon": 18.466},
    "WRO": {"name": "Wrocław Strachowice",        "city": "Wrocław",  "capacity_mpax": 3.0, "actual_mpax": 1.7, "gates": 5,  "lat": 51.102, "lon": 16.898},
    "POZ": {"name": "Poznań-Ławica",             "city": "Poznań",   "capacity_mpax": 2.5, "actual_mpax": 1.4, "gates": 4,  "lat": 52.421, "lon": 16.826},
    "RZE": {"name": "Rzeszów-Jasionka",          "city": "Rzeszów",  "capacity_mpax": 1.2, "actual_mpax": 0.5, "gates": 3,  "lat": 50.110, "lon": 22.019},
    "LCJ": {"name": "Łódź W. Reymonta",          "city": "Łódź",     "capacity_mpax": 1.5, "actual_mpax": 0.4, "gates": 3,  "lat": 51.722, "lon": 19.398},
    "SZZ": {"name": "Szczecin-Goleniów",         "city": "Szczecin", "capacity_mpax": 0.8, "actual_mpax": 0.3, "gates": 2,  "lat": 53.584, "lon": 14.902},
    "BZG": {"name": "Bydgoszcz Paderewskiego",   "city": "Bydgoszcz","capacity_mpax": 0.8, "actual_mpax": 0.3, "gates": 2,  "lat": 53.097, "lon": 17.977},
}

ROUTES = [
    ("WAW", "KRK", {"dist_km": 252, "freq_daily": 10}),
    ("WAW", "KTW", {"dist_km": 295, "freq_daily": 6}),
    ("WAW", "GDN", {"dist_km": 340, "freq_daily": 10}),
    ("WAW", "WRO", {"dist_km": 347, "freq_daily": 8}),
    ("WAW", "POZ", {"dist_km": 310, "freq_daily": 8}),
    ("WAW", "RZE", {"dist_km": 295, "freq_daily": 4}),
    ("WAW", "LCJ", {"dist_km": 121, "freq_daily": 4}),
    ("WAW", "SZZ", {"dist_km": 520, "freq_daily": 4}),
    ("WAW", "BZG", {"dist_km": 360, "freq_daily": 4}),
    ("KRK", "GDN", {"dist_km": 550, "freq_daily": 2}),
    ("KRK", "WRO", {"dist_km": 267, "freq_daily": 2}),
    ("KTW", "GDN", {"dist_km": 560, "freq_daily": 2}),
    ("GDN", "POZ", {"dist_km": 336, "freq_daily": 2}),
    ("WRO", "POZ", {"dist_km": 165, "freq_daily": 2}),
]

CRUISE_SPEED_KMH = 700
SIM_DURATION_MIN = 1440
PAX_PER_FLIGHT = 150

# Słownik na statystyki globalne symulacji SimPy
stats = {
    "flights_completed": collections.Counter(),
    "flights_delayed": collections.Counter(),
    "total_delay_min": collections.Counter(),
    "pax_handled": collections.Counter(),
    "gate_wait_times": collections.defaultdict(list),
}

# =============================================================================
# 2. BUDOWA GRAFU
# =============================================================================
def build_networkx_graph():
    G = nx.DiGraph()
    for code, data in AIRPORTS.items():
        utilization = data["actual_mpax"] / data["capacity_mpax"]
        G.add_node(
            code,
            label=data["name"],
            city=data["city"],
            capacity_mpax=data["capacity_mpax"],
            actual_mpax=data["actual_mpax"],
            utilization=round(utilization, 3),
            gates=data["gates"],
            lat=data["lat"],
            lon=data["lon"],
        )
    for src, dst, attrs in ROUTES:
        travel_min = round(attrs["dist_km"] / CRUISE_SPEED_KMH * 60)
        for a, b in [(src, dst), (dst, src)]:
            G.add_edge(
                a, b,
                dist_km=attrs["dist_km"],
                freq_daily=attrs["freq_daily"],
                travel_min=travel_min,
                weight=1.0 / attrs["freq_daily"],
            )
    return G

# =============================================================================
# 3. ANALIZA TOPOLOGICZNA (NetworkX i igraph)
# =============================================================================
def analyze_networkx(G: nx.DiGraph):
    print("\n" + "="*70)
    print("  ANALIZA TOPOLOGICZNA SIECI – NetworkX")
    print("="*70)
    print(f"\n  Węzły (lotniska)    : {G.number_of_nodes()}")
    print(f"  Krawędzie (połącz.) : {G.number_of_edges()}")
    print(f"  Gęstość grafu       : {nx.density(G):.4f}")

    degree_c   = nx.degree_centrality(G)
    between_c  = nx.betweenness_centrality(G, normalized=True, weight="weight")
    page_rank  = nx.pagerank(G, weight="freq_daily")

    print("\n  ┌─────────────────────────────────────────────────────────────────┐")
    print("  │ Centralność węzłów (ranking PageRank wg ruchu)                  │")
    print("  ├──────┬──────────────────────────┬────────┬───────────┬──────────┤")
    print("  │ IATA │ Lotnisko                 │Stopień │Pośredność │ PageRank │")
    print("  ├──────┼──────────────────────────┼────────┼───────────┼──────────┤")

    sorted_airports = sorted(page_rank, key=page_rank.get, reverse=True)
    for code in sorted_airports:
        d  = degree_c[code]
        b  = between_c[code]
        pr = page_rank[code]
        name = AIRPORTS[code]["name"][:26]
        print(f"  │ {code} │ {name:<26} │ {d:.3f} │   {b:.4f}  │  {pr:.4f}  │")
    print("  └──────┴──────────────────────────┴────────┴───────────┴──────────┘")

    print("\n  ┌────────────────────────────────────────────────────────────────┐")
    print("  │ Przepustowość i wykorzystanie portów (dane ULC 2009)           │")
    print("  ├──────┬──────────────────────────┬──────────┬─────────┬────────┤")
    print("  │ IATA │ Port                     │Przep.(M) │Ruch.(M) │Wykor.% │")
    print("  ├──────┼──────────────────────────┼──────────┼─────────┼────────┤")
    for code in sorted_airports:
        a = AIRPORTS[code]
        util_pct = a["actual_mpax"] / a["capacity_mpax"] * 100
        name = a["name"][:26]
        print(f"  │ {code} │ {name:<26} │   {a['capacity_mpax']:5.1f}  │  {a['actual_mpax']:5.1f}  │ {util_pct:5.1f}% │")
    print("  └──────┴──────────────────────────┴──────────┴─────────┴────────┘")

    G_undir = G.to_undirected()
    print(f"\n  Spójność grafu         : {'TAK' if nx.is_connected(G_undir) else 'NIE'}")
    try:
        avg_path = nx.average_shortest_path_length(G_undir, weight="weight")
        print(f"  Śr. długość ścieżki    : {avg_path:.3f} (wg wagi 1/freq)")
    except Exception:
        print("  Śr. długość ścieżki    : (graf niespójny)")

    path = nx.shortest_path(G, "WAW", "RZE", weight=None)
    print(f"\n  Najkrótsza trasa WAW → RZE (min. przesiadek): {' → '.join(path)}")

    flow_val, _ = nx.maximum_flow(G, "WAW", "KRK", capacity="freq_daily")
    print(f"  Max-flow WAW → KRK     : {flow_val:.0f} lotów/dobę")

    return page_rank, between_c

def analyze_igraph(G_nx: nx.DiGraph):
    print("\n" + "="*70)
    print("  ANALIZA GRAFU – igraph (community detection, clustering)")
    print("="*70)

    nodes = list(G_nx.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges_ig = [(node_idx[u], node_idx[v]) for u, v in G_nx.edges()]
    weights_ig = [G_nx[u][v]["freq_daily"] for u, v in G_nx.edges()]

    g = ig.Graph(n=len(nodes), edges=edges_ig, directed=True)
    g.vs["name"] = nodes
    g.vs["label"] = [AIRPORTS[n]["city"] for n in nodes]
    g.es["weight"] = weights_ig

    g_undir = g.as_undirected(combine_edges="sum")
    print("\n  Współczynnik klasteryzacji (lokalny) – igraph:")
    clustering = g_undir.transitivity_local_undirected(mode="zero")
    for i, v in enumerate(g_undir.vs):
        print(f"    {nodes[i]} ({AIRPORTS[nodes[i]]['city']:<15}): {clustering[i]:.4f}")

    global_clust = g_undir.transitivity_undirected()
    print(f"\n  Globalny wsp. klasteryzacji: {global_clust:.4f}")

    pr_ig = g.pagerank(weights="weight")
    print("\n  PageRank (igraph, wg freq_daily):")
    pr_sorted = sorted(zip(nodes, pr_ig), key=lambda x: -x[1])
    for code, pr in pr_sorted:
        bar = "█" * int(pr * 200)
        print(f"    {code}: {pr:.5f}  {bar}")

    try:
        communities = g_undir.community_multilevel(weights="weight")
        print(f"\n  Detekcja społeczności (Louvain): {len(communities)} grupy/grup")
        for i, comm in enumerate(communities):
            members = [nodes[j] for j in comm]
            print(f"    Grupa {i+1}: {members}")
        print(f"  Modularność: {communities.modularity:.4f}")
    except Exception as e:
        print(f"  Community detection: {e}")

    try:
        diam = g_undir.diameter(weights=None)
        print(f"\n  Średnica grafu (hop distance): {diam}")
    except Exception:
        pass

# =============================================================================
# 4. SYMULACJA RUCHU – SimPy
# =============================================================================
def flight_process(env, flight_id, src, dst, travel_min, gate_resources):
    pax = random.randint(int(PAX_PER_FLIGHT * 0.7), int(PAX_PER_FLIGHT * 1.1))

    # --- DEPARTURE ---
    departure_request_time = env.now
    with gate_resources[src].request() as req:
        yield req
        gate_wait = env.now - departure_request_time
        if gate_wait > 0:
            stats["flights_delayed"][src] += 1
            stats["total_delay_min"][src] += gate_wait
        stats["gate_wait_times"][src].append(gate_wait)

        boarding_time = random.normalvariate(25, 5)
        yield env.timeout(max(boarding_time, 10))

    stats["pax_handled"][src] += pax

    # --- FLIGHT ---
    actual_travel = travel_min * random.uniform(0.9, 1.1)
    yield env.timeout(actual_travel)

    # --- ARRIVAL ---
    arrival_request_time = env.now
    with gate_resources[dst].request() as req:
        yield req
        arr_gate_wait = env.now - arrival_request_time
        if arr_gate_wait > 0:
            stats["flights_delayed"][dst] += 1
            stats["total_delay_min"][dst] += arr_gate_wait
        stats["gate_wait_times"][dst].append(arr_gate_wait)

        deboard_time = random.normalvariate(20, 4)
        yield env.timeout(max(deboard_time, 10))

    stats["pax_handled"][dst] += pax
    stats["flights_completed"][f"{src}→{dst}"] += 1

def route_generator(env, src, dst, travel_min, freq_daily, gate_resources):
    mean_interarrival = SIM_DURATION_MIN / (freq_daily / 2)
    flight_counter = itertools.count(1)
    while True:
        interarrival = random.expovariate(1 / mean_interarrival)
        yield env.timeout(interarrival)
        if env.now >= SIM_DURATION_MIN:
            break
        fid = f"{src}{dst}_{next(flight_counter):04d}"
        env.process(flight_process(env, fid, src, dst, travel_min, gate_resources))

def run_simulation():
    print("\n" + "="*70)
    print("  SYMULACJA RUCHU LOTNICZEGO – SimPy (1 doba = 1440 min)")
    print("="*70)

    env = simpy.Environment()
    gate_resources = {
        code: simpy.Resource(env, capacity=data["gates"])
        for code, data in AIRPORTS.items()
    }

    for src, dst, attrs in ROUTES:
        travel_min = round(attrs["dist_km"] / CRUISE_SPEED_KMH * 60)
        for a, b in [(src, dst), (dst, src)]:
            env.process(route_generator(env, a, b, travel_min, attrs["freq_daily"], gate_resources))

    env.run(until=SIM_DURATION_MIN)
    print(f"\n  Symulacja zakończona. Czas: {SIM_DURATION_MIN} min (1 doba)")

    print("\n  ── Ukończone loty wg tras ──────────────────────────────────────")
    total_flights = 0
    for route, count in sorted(stats["flights_completed"].items(), key=lambda x: -x[1]):
        total_flights += count
        print(f"    {route}: {count:3d} lotów")
    print(f"    ───────────────────────────────")
    print(f"    RAZEM: {total_flights} lotów")

    print("\n  ── Statystyki lotnisk (wyniki symulacji) ───────────────────────")
    print(f"  {'IATA':<6} {'Miasto':<15} {'Pax':<8} {'Op.opóźn.':<12} {'Śr.opóźn.':<12} {'Wykor.%'}")
    print("  " + "─"*66)

    for code, data in AIRPORTS.items():
        pax = stats["pax_handled"].get(code, 0)
        delayed = stats["flights_delayed"].get(code, 0)
        wait_times = stats["gate_wait_times"].get(code, [0])
        avg_wait = np.mean(wait_times) if wait_times else 0.0

        daily_cap = data["capacity_mpax"] * 1_000_000 / 365
        sim_util = min(pax / daily_cap * 100, 100) if daily_cap > 0 else 0

        print(f"  {code:<6} {data['city']:<15} {pax:>7,}  {delayed:>10}  {avg_wait:>9.1f} min  {sim_util:>6.1f}%")

def complex_network_metrics(G_nx: nx.DiGraph, between_c: dict):
    print("\n" + "="*70)
    print("  METRYKI SIECI ZŁOŻONEJ")
    print("="*70)

    G_undir = G_nx.to_undirected()
    degrees = dict(G_nx.degree())
    deg_values = list(degrees.values())
    print(f"\n  Rozkład stopni (degree distribution):")
    print(f"    Min stopień  : {min(deg_values)}")
    print(f"    Max stopień  : {max(deg_values)}")
    print(f"    Śr. stopień  : {np.mean(deg_values):.2f}")
    print(f"    Odch. std.   : {np.std(deg_values):.2f}")

    cv = np.std(deg_values) / np.mean(deg_values)
    print(f"    Wsp. zmienności: {cv:.3f} ", end="")
    if cv > 0.5:
        print("→ topologia HUB-and-SPOKE (WAW dominuje jako hub)")
    else:
        print("→ sieć bardziej jednorodna")

    bottleneck = max(between_c, key=between_c.get)
    print(f"\n  Węzeł krytyczny (max betweenness): {bottleneck} ({AIRPORTS[bottleneck]['city']}) = {between_c[bottleneck]:.4f}")
    print(f"  → Awaria {bottleneck} najbardziej zakłóci sieć krajową")

    print(f"\n  Test odporności (usuwanie węzłów wg betweenness):")
    G_test = G_undir.copy()
    sorted_nodes = sorted(between_c, key=between_c.get, reverse=True)
    for node in sorted_nodes:
        if not nx.is_connected(G_test):
            break
        G_test.remove_node(node)
        connected = nx.is_connected(G_test)
        print(f"    Usunięto {node}: sieć {'spójna ✓' if connected else 'NIESPÓJNA ✗'}")
        if not connected:
            break

    try:
        avg_l = nx.average_shortest_path_length(G_undir, weight=None)
        n = G_undir.number_of_nodes()
        m = G_undir.number_of_edges()
        p = m / (n * (n - 1) / 2)
        l_rand = np.log(n) / np.log(n * p) if n * p > 1 else float("inf")
        c_real = nx.transitivity(G_undir)
        c_rand = p
        print(f"\n  Właściwości Small-World:")
        print(f"    L (śr. ścieżka, rzeczyw.) : {avg_l:.3f}")
        print(f"    L (oczekiwana, losowa)    : {l_rand:.3f}")
        print(f"    C (klasteryzacja, rzeczyw.): {c_real:.4f}")
        print(f"    C (oczekiwana, losowa)    : {c_rand:.4f}")
        sw_sigma = (c_real / c_rand) / (avg_l / l_rand)
        print(f"    σ (Small-World index)     : {sw_sigma:.3f}", end="")
        print(" → sieć Small-World ✓" if sw_sigma > 1 else " → brak efektu Small-World")
    except Exception as e:
        print(f"  Small-World: {e}")

# =============================================================================
# 5. GENEROWANIE WYKRESÓW (WIZUALIZACJA)
# =============================================================================
def plot_network_map(G, page_rank, ax):
    ax.set_facecolor("#0d1b2a")
    pos = {code: (data["lon"], data["lat"]) for code, data in AIRPORTS.items()}

    edges_seen = set()
    for u, v, d in G.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in edges_seen:
            continue
        edges_seen.add(key)
        freq = d["freq_daily"]
        lw = 0.5 + freq * 0.25
        alpha = 0.3 + freq * 0.04
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color="#4fc3f7", linewidth=lw, alpha=min(alpha, 0.85), zorder=1)

        if freq >= 8:
            mx, my = (x[0]+x[1])/2, (y[0]+y[1])/2
            ax.text(mx, my, f"{freq}×", fontsize=5.5, color="#b0bec5",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#0d1b2a", ec="none", alpha=0.7))

    codes = list(AIRPORTS.keys())
    pr_vals = np.array([page_rank[c] for c in codes])
    sizes = 80 + pr_vals / pr_vals.max() * 900

    cap_vals = np.array([AIRPORTS[c]["capacity_mpax"] for c in codes])
    norm = mcolors.Normalize(vmin=cap_vals.min(), vmax=cap_vals.max())
    cmap = plt.cm.YlOrRd

    xs = [pos[c][0] for c in codes]
    ys = [pos[c][1] for c in codes]
    sc = ax.scatter(xs, ys, s=sizes, c=cap_vals, cmap=cmap, norm=norm,
                    zorder=3, edgecolors="#ffffff", linewidths=0.8)

    offsets = {
        "WAW": (0.35, 0.1), "KRK": (0.35, -0.18), "KTW": (-0.6, -0.2),
        "GDN": (0.35, 0.1), "WRO": (-0.6, 0.05),  "POZ": (-0.55, 0.1),
        "RZE": (0.35, 0.05),"LCJ": (-0.55, -0.18),"SZZ": (-0.6, 0.1),
        "BZG": (0.35, 0.1),
    }
    for code in codes:
        ox, oy = offsets.get(code, (0.3, 0.1))
        ax.annotate(code, xy=(pos[code][0], pos[code][1]), xytext=(pos[code][0]+ox, pos[code][1]+oy),
                    fontsize=8, fontweight="bold", color="white", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#78909c", lw=0.6))

    cbar = plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Przepustowość (mln pax/rok)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    for freq, label in [(4, "4 loty/dobę"), (8, "8 lotów/dobę"), (10, "10 lotów/dobę")]:
        ax.plot([], [], color="#4fc3f7", linewidth=0.5+freq*0.25, label=label, alpha=0.8)
    ax.legend(fontsize=7, loc="lower right", facecolor="#1a2a3a", labelcolor="white", edgecolor="#4fc3f7", framealpha=0.9)

    ax.set_xlabel("Długość geograficzna", color="white", fontsize=8)
    ax.set_ylabel("Szerokość geograficzna", color="white", fontsize=8)
    ax.set_title("Sieć połączeń lotniczych w Polsce\n(rozmiar węzła = PageRank, grubość = częstotliwość)",
                 color="white", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#37474f")

def plot_degree_distribution(G, ax):
    ax.set_facecolor("#0d1b2a")
    degrees = dict(G.degree())
    codes = list(degrees.keys())
    deg_vals = [degrees[c] for c in codes]
    cities = [AIRPORTS[c]["city"] for c in codes]

    order = sorted(range(len(codes)), key=lambda i: -deg_vals[i])
    codes_s   = [codes[i] for i in order]
    deg_s     = [deg_vals[i] for i in order]
    cities_s  = [cities[i] for i in order]

    colors = ["#ef5350" if codes_s[i] == "WAW" else "#42a5f5" for i in range(len(codes_s))]
    bars = ax.bar(range(len(codes_s)), deg_s, color=colors, edgecolor="#37474f", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, deg_s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(codes_s)))
    ax.set_xticklabels([f"{c}\n{cities_s[i]}" for i, c in enumerate(codes_s)], fontsize=8, color="white")
    ax.set_ylabel("Stopień węzła (in + out)", color="white", fontsize=9)
    ax.set_title("Rozkład stopni węzłów sieci\n(stopień = sumaryczna liczba połączeń)", color="white", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    ax.set_ylim(0, max(deg_s) * 1.15)
    for spine in ax.spines.values():
        spine.set_edgecolor("#37474f")
    ax.grid(axis="y", color="#263238", linewidth=0.8, zorder=0)

    hub_patch  = mpatches.Patch(color="#ef5350", label="Hub (WAW)")
    node_patch = mpatches.Patch(color="#42a5f5", label="Węzeł regionalny")
    ax.legend(handles=[hub_patch, node_patch], fontsize=8, facecolor="#1a2a3a", labelcolor="white", edgecolor="#4fc3f7")

def plot_simulation_heatmap(sim_stats, ax):
    ax.set_facecolor("#0d1b2a")
    codes = list(AIRPORTS.keys())
    cities = [AIRPORTS[c]["city"] for c in codes]

    pax_vals = np.array([sim_stats["pax_handled"].get(c, 0) for c in codes], dtype=float)
    delay_vals = np.array([sim_stats["flights_delayed"].get(c, 0) for c in codes], dtype=float)
    avg_wait = np.array([np.mean(sim_stats["gate_wait_times"].get(c, [0])) for c in codes], dtype=float)

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    data_matrix = np.vstack([norm01(pax_vals), norm01(delay_vals), norm01(avg_wait)])
    row_labels = ["Pasażerowie\n(znorm.)", "Opóźnione loty\n(znorm.)", "Śr. czas czekania\n(znorm.)"]

    im = ax.imshow(data_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([f"{c}\n{cities[i]}" for i, c in enumerate(codes)], fontsize=8, color="white")
    
    # Naprawiono błąd: Zmieniono z set_yaxis na set_yticks
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, color="white")

    raw_rows = [pax_vals, delay_vals, avg_wait]
    fmts     = ["{:,.0f}", "{:.0f}", "{:.1f}m"]
    for row_i, (raw, fmt) in enumerate(zip(raw_rows, fmts)):
        for col_i, val in enumerate(raw):
            txt_color = "black" if data_matrix[row_i, col_i] > 0.5 else "white"
            ax.text(col_i, row_i, fmt.format(val), ha="center", va="center", fontsize=7, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Intensywność (0=min, 1=max)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    ax.set_title("Heatmapa obciążenia lotnisk – wyniki symulacji (1 doba)", color="white", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#37474f")

def generate_visualizations(G, page_rank, sim_stats, output_path="wizualizacja_1.png"):
    print("\n" + "="*70)
    print("  GENEROWANIE WYKRESÓW – Matplotlib")
    print("="*70)
    
    fig = plt.figure(figsize=(20, 16), facecolor="#0a1520")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32, left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_map  = fig.add_subplot(gs[0, :])
    ax_deg  = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    plot_network_map(G, page_rank, ax_map)
    plot_degree_distribution(G, ax_deg)
    plot_simulation_heatmap(sim_stats, ax_heat)

    fig.suptitle("SYMULACJA SIECI ZŁOŻONEJ LOTNISK W POLSCE", fontsize=15, fontweight="bold", color="white", y=0.97)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Sukces! Wykresy zapisano w pliku: {output_path}")

# =============================================================================
# 6. GŁÓWNY PUNKT WEJŚCIA (MAIN)
# =============================================================================
def main():
    print("\n" + "█"*70)
    print("  ZINTEGROWANY PROJEKT: ANALIZA I SYMULACJA POLSKICH PORTÓW")
    print("█"*70)

    # 1. Budowa grafu
    G = build_networkx_graph()

    # 2. Analiza topologiczna NetworkX
    page_rank, bet_c = analyze_networkx(G)

    # 3. Analiza społeczności igraph
    analyze_igraph(G)

    # 4. Uruchomienie symulacji SimPy (Statystyki lądują w globalnym słowniku `stats`)
    run_simulation()

    # 5. Wyznaczenie zaawansowanych metryk sieci złożonej
    complex_network_metrics(G, bet_c)

    # 6. Generowanie i zapis wizualizacji (przekazujemy wyniki z jednego przebiegu)
    generate_visualizations(G, page_rank, stats, output_path="wizualizacja_symulacji.png")

    print("\n" + "█"*70)
    print("  WSZYSTKIE PROCESY ZAKOŃCZONE POMYŚLNIE")
    print("█"*70 + "\n")

if __name__ == "__main__":
    main()