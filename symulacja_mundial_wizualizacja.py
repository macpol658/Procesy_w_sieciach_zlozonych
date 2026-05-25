"""
=============================================================================
ZINTEGROWANA SYMULACJA SIECI ZŁOŻONEJ – SCENARIUSZ MUNDIAL (WZMOŻONY RUCH)
=============================================================================
Scenariusz: Polska organizuje turniej mistrzowski. Wybrane porty regionalne
stają się miastami-gospodarzami, co generuje ekstremalny ruch wahadłowy.
Symulacja bada odporność infrastruktury (kolejki do bramek) przy przeciążeniu.
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")  # Tryb bezokienkowy do bezpośredniego zapisu plików
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import simpy
import networkx as nx
import random
import numpy as np
import collections
import itertools

# ── Deterministyczne ziarno dla powtarzalności ──────────────────────────────
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DANE WEJŚCIOWE I KONFIGURACJA SCENARIUSZA MUNDIALOWEGO
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

# Bazowe trasy przed modyfikacją mundialową
BASE_ROUTES = [
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

HOST_AIRPORTS = {"WAW", "KRK", "GDN", "WRO"}  # Lotniska obsługujące stadiony

# Generowanie tras ze zmodyfikowanym, wzmożonym ruchem dla miast-gospodarzy
MUNDIAL_ROUTES = []
for src, dst, attrs in BASE_ROUTES:
    freq = attrs["freq_daily"]
    # Jeśli trasa łączy się bezpośrednio z miastem gospodarzem, potrajamy lub podwajamy ruch kibiców
    if src in HOST_AIRPORTS or dst in HOST_AIRPORTS:
        freq = int(freq * 3.5) if (src == "WAW" or dst == "WAW") else int(freq * 2.5)
    
    MUNDIAL_ROUTES.append((src, dst, {"dist_km": attrs["dist_km"], "freq_daily": freq}))

CRUISE_SPEED_KMH = 700
SIM_DURATION_MIN = 1440
PAX_PER_FLIGHT = 165  # Większe samoloty (czartery) podstawione na turniej

stats = {
    "flights_completed": collections.Counter(),
    "flights_delayed": collections.Counter(),
    "total_delay_min": collections.Counter(),
    "pax_handled": collections.Counter(),
    "gate_wait_times": collections.defaultdict(list),
}

# =============================================================================
# 2. BUDOWA GRAFU MUNDIALOWEGO
# =============================================================================
def build_networkx_graph():
    G = nx.DiGraph()
    for code, data in AIRPORTS.items():
        G.add_node(
            code,
            label=data["name"],
            city=data["city"],
            capacity_mpax=data["capacity_mpax"],
            gates=data["gates"],
            lat=data["lat"],
            lon=data["lon"],
            is_host=(code in HOST_AIRPORTS)
        )
    for src, dst, attrs in MUNDIAL_ROUTES:
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
# 3. SYMULACJA RUCHU (SimPy)
# =============================================================================
def flight_process(env, flight_id, src, dst, travel_min, gate_resources):
    pax = random.randint(int(PAX_PER_FLIGHT * 0.85), int(PAX_PER_FLIGHT * 1.15))

    # --- DEPARTURE ---
    departure_request_time = env.now
    with gate_resources[src].request() as req:
        yield req
        gate_wait = env.now - departure_request_time
        if gate_wait > 0:
            stats["flights_delayed"][src] += 1
            stats["total_delay_min"][src] += gate_wait
        stats["gate_wait_times"][src].append(gate_wait)

        boarding_time = random.normalvariate(28, 4)
        yield env.timeout(max(boarding_time, 10))

    stats["pax_handled"][src] += pax

    # --- FLIGHT ---
    actual_travel = travel_min * random.uniform(0.95, 1.05)
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

        deboard_time = random.normalvariate(22, 3)
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
    print("  URUCHAMIANIE SYMULACJI KRYZYSOWEJ: MUNDIAL (Ekstremalny popyt)")
    print("="*70)

    env = simpy.Environment()
    gate_resources = {code: simpy.Resource(env, capacity=data["gates"]) for code, data in AIRPORTS.items()}

    for src, dst, attrs in MUNDIAL_ROUTES:
        travel_min = round(attrs["dist_km"] / CRUISE_SPEED_KMH * 60)
        for a, b in [(src, dst), (dst, src)]:
            env.process(route_generator(env, a, b, travel_min, attrs["freq_daily"], gate_resources))

    env.run(until=SIM_DURATION_MIN)
    print(f"\n  Symulacja zakończona. Czas turniejowy: {SIM_DURATION_MIN} min (1 doba)")

    print("\n  ── Obciążenie lotnisk w warunkach turnieju ─────────────────────")
    print(f"  {'IATA':<6} {'Status':<12} {'Pax obsłuż.'} {'Loty opóźn.'} {'Śr. oczekiwanie bramka'}")
    print("  " + "─"*66)

    for code, data in AIRPORTS.items():
        pax = stats["pax_handled"].get(code, 0)
        delayed = stats["flights_delayed"].get(code, 0)
        wait_times = stats["gate_wait_times"].get(code, [0])
        avg_wait = np.mean(wait_times) if wait_times else 0.0
        status_str = "GOSPODARZ" if code in HOST_AIRPORTS else "Regionalne"

        print(f"  {code:<6} {status_str:<12} {pax:>11,} {delayed:>12} {avg_wait:>18.1f} min")

# =============================================================================
# 4. WIZUALIZACJA ZACHOWANIA SIECI PRZY PRZECIĄŻENIU
# =============================================================================
def plot_network_map(G, page_rank, ax):
    ax.set_facecolor("#111625")
    pos = {code: (data["lon"], data["lat"]) for code, data in AIRPORTS.items()}

    edges_seen = set()
    for u, v, d in G.edges(data=True):
        key = tuple(sorted([u, v]))
        if key in edges_seen:
            continue
        edges_seen.add(key)
        freq = d["freq_daily"]
        lw = 0.5 + freq * 0.12  # Skalowanie grubości krawędzi
        alpha = 0.2 + min(freq * 0.015, 0.65)
        
        # Kolor krawędzi - wyróżnienie połączeń ze strefami gospodarzy
        edge_color = "#ff3366" if (u in HOST_AIRPORTS or v in HOST_AIRPORTS) else "#4fc3f7"
        
        # Dodano brakujące mapowanie wektorów położenia geograficznego
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        ax.plot(x, y, color=edge_color, linewidth=lw, alpha=alpha, zorder=1)

        if freq >= 15:
            mx, my = (pos[u][0]+pos[v][0])/2, (pos[u][1]+pos[v][1])/2
            ax.text(mx, my, f"{freq}×", fontsize=6, color="#ffffff", ha="center", va="center", zorder=5)

    codes = list(AIRPORTS.keys())
    pr_vals = np.array([page_rank[c] for c in codes])
    sizes = 100 + pr_vals / pr_vals.max() * 1200

    node_colors = ["#ff3366" if c in HOST_AIRPORTS else "#42a5f5" for c in codes]
    xs = [pos[c][0] for c in codes]
    ys = [pos[c][1] for c in codes]
    
    ax.scatter(xs, ys, s=sizes, c=node_colors, zorder=3, edgecolors="#ffffff", linewidths=1.2)

    for code in codes:
        ax.annotate(code, xy=(pos[code][0], pos[code][1]), xytext=(pos[code][0]+0.3, pos[code][1]+0.1),
                    fontsize=9, fontweight="bold", color="white", zorder=6)

    ax.set_title("Struktura potoków pasażerskich (Mundial)\n[Czerwone krawędzie i węzły = Przeciążone sektory gospodarzy]",
                 color="white", fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#263238")

def plot_degree_distribution(G, ax):
    ax.set_facecolor("#111625")
    degrees = dict(G.degree())
    codes = list(degrees.keys())
    deg_vals = [degrees[c] for c in codes]

    order = sorted(range(len(codes)), key=lambda i: -deg_vals[i])
    codes_s   = [codes[i] for i in order]
    deg_s     = [deg_vals[i] for i in order]

    colors = ["#ff3366" if c in HOST_AIRPORTS else "#42a5f5" for c in codes_s]
    bars = ax.bar(range(len(codes_s)), deg_s, color=colors, edgecolor="#37474f", linewidth=0.8, zorder=3)

    for bar, val in zip(bars, deg_s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(val), ha="center", va="bottom", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(codes_s)))
    ax.set_xticklabels(codes_s, fontsize=9, color="white")
    ax.set_ylabel("Stopień węzła sieci", color="white", fontsize=9)
    ax.set_title("Topologiczny stopień węzłów w sieci połączeń", color="white", fontsize=11, fontweight="bold", pad=10)
    ax.grid(axis="y", color="#263238", linewidth=0.8, zorder=0)

def plot_simulation_heatmap(sim_stats, ax):
    ax.set_facecolor("#111625")
    codes = list(AIRPORTS.keys())
    cities = [AIRPORTS[c]["city"] for c in codes]

    pax_vals = np.array([sim_stats["pax_handled"].get(c, 0) for c in codes], dtype=float)
    delay_vals = np.array([sim_stats["flights_delayed"].get(c, 0) for c in codes], dtype=float)
    avg_wait = np.array([np.mean(sim_stats["gate_wait_times"].get(c, [0])) for c in codes], dtype=float)

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    data_matrix = np.vstack([norm01(pax_vals), norm01(delay_vals), norm01(avg_wait)])
    row_labels = ["Pasażerowie\n(znorm.)", "Opóźnione starty\n(znorm.)", "Kolejka Gate\n(znorm.)"]

    im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([f"{c}\n{cities[i]}" for i, c in enumerate(codes)], fontsize=8, color="white")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, color="white")

    raw_rows = [pax_vals, delay_vals, avg_wait]
    fmts     = ["{:,.0f}", "{:.0f}", "{:.1f}m"]
    for row_i, (raw, fmt) in enumerate(zip(raw_rows, fmts)):
        for col_i, val in enumerate(raw):
            txt_color = "black" if data_matrix[row_i, col_i] > 0.4 else "white"
            ax.text(col_i, row_i, fmt.format(val), ha="center", va="center", fontsize=7.5, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    ax.set_title("Wąskie gardła i wydolność węzłów (Heatmapa obciążenia)", color="white", fontsize=11, fontweight="bold", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#263238")

def generate_visualizations(G, page_rank, sim_stats, output_path="wizualizacja_mundial.png"):
    print("\n  [VIZ] Generowanie map obciążeń kryzysowych...")
    fig = plt.figure(figsize=(22, 16), facecolor="#090d16")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32, left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_map  = fig.add_subplot(gs[0, :])
    ax_deg  = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    plot_network_map(G, page_rank, ax_map)
    plot_degree_distribution(G, ax_deg)
    plot_simulation_heatmap(sim_stats, ax_heat)

    fig.suptitle("REAKCJA SIECI NA NAGŁY WZROST OBLOTÓW (SCENARIUSZ: MUNDIAL)", fontsize=16, fontweight="bold", color="white", y=0.97)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Analiza wizualna zapisana pomyślnie w: {output_path}")

# =============================================================================
# 5. GŁÓWNY PUNKT URUCHOMIENIA
# =============================================================================
def main():
    run_simulation()
    G = build_networkx_graph()
    page_rank = nx.pagerank(G, weight="freq_daily")
    generate_visualizations(G, page_rank, stats, output_path="wizualizacja_mundial.png")

if __name__ == "__main__":
    main()