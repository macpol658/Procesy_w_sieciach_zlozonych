"""
=============================================================================
ZINTEGROWANA SYMULACJA POGODOWA SIECI ZŁOŻONEJ LOTNISK W POLSCE
=============================================================================
Źródło danych: ULC, "Szacowane przepustowości portów lotniczych", lipiec 2009
Funkcje kryzysowe: Ground Delay Program, Diversions (Haversine), Wizualizacja
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
import math

# ── Deterministyczne ziarno dla powtarzalności ──────────────────────────────
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DANE WEJŚCIOWE I ZMIENNE GLOBALNE
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

airport_is_open = {code: True for code in AIRPORTS.keys()}

stats = {
    "flights_completed": collections.Counter(),
    "flights_delayed": collections.Counter(),
    "total_delay_min": collections.Counter(),
    "pax_handled": collections.Counter(),
    "gate_wait_times": collections.defaultdict(list),
    "diverted_flights": collections.Counter(),
    "weather_delays_min": collections.Counter(),
}

# =============================================================================
# 2. LOGIKA GEOGRAFICZNA (HAVERSINE)
# =============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_nearest_open_airport(closed_airport_code, origin_airport_code):
    best_airport = None
    min_dist = float('inf')
    lat_target = AIRPORTS[closed_airport_code]["lat"]
    lon_target = AIRPORTS[closed_airport_code]["lon"]

    for code, is_open in airport_is_open.items():
        if is_open and code != closed_airport_code and code != origin_airport_code:
            lat_cand = AIRPORTS[code]["lat"]
            lon_cand = AIRPORTS[code]["lon"]
            dist = haversine_distance(lat_target, lon_target, lat_cand, lon_cand)
            if dist < min_dist:
                min_dist = dist
                best_airport = code
    return best_airport, min_dist

# =============================================================================
# 3. PROCESY SIMPY (Z ZARZĄDZANIEM KRYZYSOWYM)
# =============================================================================
def weather_disruption_process(env, target_airport, start_time_min, duration_min):
    yield env.timeout(start_time_min)
    print(f"\n[Czas: {env.now:>4.0f} min] ⚠️ ALARM POGODOWY: Lotnisko {target_airport} zostaje ZAMKNIĘTE (np. mgła)!")
    airport_is_open[target_airport] = False

    yield env.timeout(duration_min)
    print(f"\n[Czas: {env.now:>4.0f} min] ☀️ POPRAWA POGODY: Lotnisko {target_airport} ponownie OTWARTE.")
    airport_is_open[target_airport] = True

def flight_process(env, flight_id, src, dst, travel_min, gate_resources):
    pax = random.randint(int(PAX_PER_FLIGHT * 0.7), int(PAX_PER_FLIGHT * 1.1))
    departure_request_time = env.now

    # 1. GROUND DELAY PROGRAM
    while not airport_is_open[dst]:
        yield env.timeout(10)
        stats["weather_delays_min"][src] += 10

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

    # --- LOT W POWIETRZU ---
    actual_travel = travel_min * random.uniform(0.9, 1.1)
    yield env.timeout(actual_travel)

    # --- ARRIVAL (ZBLIŻANIE) ---
    original_dst = dst

    # 2. DIVERSION
    if not airport_is_open[dst]:
        nearest_airport, extra_dist_km = get_nearest_open_airport(dst, src)
        if nearest_airport:
            dst = nearest_airport
            extra_travel_min = extra_dist_km / (CRUISE_SPEED_KMH / 60)
            print(f"[Czas: {env.now:>4.0f} min] 🔀 PRZEKIEROWANIE: Lot {flight_id} ({src}→{original_dst}). Cel ZAMKNIĘTY. Leci do: {dst} (+{extra_dist_km:.0f} km / +{extra_travel_min:.0f} min).")
            stats["diverted_flights"][original_dst] += 1
            yield env.timeout(extra_travel_min)
        else:
            print(f"[Czas: {env.now:>4.0f} min] ⚠️ KRYZYS: Lot {flight_id} nie ma otwartej alternatywy! Wraca do {src}.")
            dst = src
            yield env.timeout(actual_travel)

    # --- LĄDOWANIE ---
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
    route_key = f"{src}→{dst}" if original_dst == dst else f"{src}→{dst} (Przekierowano z {original_dst})"
    stats["flights_completed"][route_key] += 1

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
    print("\n" + "=" * 70)
    print("  SYMULACJA RUCHU LOTNICZEGO – SimPy (z analizą zakłóceń pogodowych)")
    print("=" * 70)

    env = simpy.Environment()
    gate_resources = {code: simpy.Resource(env, capacity=data["gates"]) for code, data in AIRPORTS.items()}

    for src, dst, attrs in ROUTES:
        travel_min = round(attrs["dist_km"] / CRUISE_SPEED_KMH * 60)
        for a, b in [(src, dst), (dst, src)]:
            env.process(route_generator(env, a, b, travel_min, attrs["freq_daily"], gate_resources))

    # Symulacja zamknięcia WAW we mgle na 3 godziny (12:00 - 15:00)
    env.process(weather_disruption_process(env, target_airport="WAW", start_time_min=720, duration_min=180))
    env.run(until=SIM_DURATION_MIN)

    print("\n" + "=" * 70)
    print(f"  WYNIKI SYMULACJI (Czas: {SIM_DURATION_MIN} min)")
    print("=" * 70)
    print("\n  ── Statystyki lotnisk ───────────────────────────────────────────────")
    print(f"  {'IATA':<5} {'Miasto':<12} {'Pax':<6} {'Op.Gate':<8} {'Śr.Gate':<8} {'Przek.':<7} {'Wstrzymane (min)'}")
    print("  " + "─" * 72)

    for code, data in AIRPORTS.items():
        pax = stats["pax_handled"].get(code, 0)
        delayed = stats["flights_delayed"].get(code, 0)
        wait_times = stats["gate_wait_times"].get(code, [0])
        avg_wait = np.mean(wait_times) if wait_times else 0.0
        diverts = stats["diverted_flights"].get(code, 0)
        weather_wait = stats["weather_delays_min"].get(code, 0)

        print(f"  {code:<5} {data['city']:<12} {pax:>6,}  {delayed:>7}  {avg_wait:>7.1f}m  {diverts:>6}  {weather_wait:>12}m")

# =============================================================================
# 4. BUDOWA GRAFU I ANALIZA TOPOLOGICZNA (DLA WIZUALIZACJI)
# =============================================================================
def build_graph():
    G = nx.DiGraph()
    for code, data in AIRPORTS.items():
        G.add_node(code, lat=data["lat"], lon=data["lon"])
    for src, dst, attrs in ROUTES:
        for a, b in [(src, dst), (dst, src)]:
            G.add_edge(a, b, dist_km=attrs["dist_km"], freq_daily=attrs["freq_daily"], weight=1.0 / attrs["freq_daily"])
    return G

# =============================================================================
# 5. GENEROWANIE WYKRESÓW
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
            ax.text(mx, my, f"{freq}×", fontsize=5.5, color="#b0bec5", ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#0d1b2a", ec="none", alpha=0.7))

    codes = list(AIRPORTS.keys())
    pr_vals = np.array([page_rank[c] for c in codes])
    sizes = 80 + pr_vals / pr_vals.max() * 900

    cap_vals = np.array([AIRPORTS[c]["capacity_mpax"] for c in codes])
    norm = mcolors.Normalize(vmin=cap_vals.min(), vmax=cap_vals.max())
    cmap = plt.cm.YlOrRd

    xs = [pos[c][0] for c in codes]
    ys = [pos[c][1] for c in codes]
    sc = ax.scatter(xs, ys, s=sizes, c=cap_vals, cmap=cmap, norm=norm, zorder=3, edgecolors="#ffffff", linewidths=0.8)

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
    ax.set_title("Rozkład stopni węzłów sieci\n(stopień = sumaryczna liczba połączeń wejściowych + wyjściowych)",
                 color="white", fontsize=10, fontweight="bold", pad=10)
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
    divert_vals = np.array([sim_stats["diverted_flights"].get(c, 0) for c in codes], dtype=float)
    gdp_vals = np.array([sim_stats["weather_delays_min"].get(c, 0) for c in codes], dtype=float)

    def norm01(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    data_matrix = np.vstack([norm01(pax_vals), norm01(divert_vals), norm01(gdp_vals)])
    row_labels = ["Pasażerowie\n(znorm.)", "Przekierowania\n(znorm.)", "Ground Delay\n(znorm.)"]

    im = ax.imshow(data_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([f"{c}\n{cities[i]}" for i, c in enumerate(codes)], fontsize=8, color="white")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9, color="white")

    raw_rows = [pax_vals, divert_vals, gdp_vals]
    fmts     = ["{:,.0f}", "{:.0f}", "{:.0f}m"]
    for row_i, (raw, fmt) in enumerate(zip(raw_rows, fmts)):
        for col_i, val in enumerate(raw):
            txt_color = "black" if data_matrix[row_i, col_i] > 0.5 else "white"
            ax.text(col_i, row_i, fmt.format(val), ha="center", va="center", fontsize=7, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Intensywność (0=min, 1=max)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    ax.set_title("Heatmapa kryzysowa – wyniki wpływu pogody (1 doba)", color="white", fontsize=10, fontweight="bold", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#37474f")

def generate_visualizations(G, page_rank, sim_stats, output_path="wizualizacja_pogoda.png"):
    print("\n  [VIZ] Generuję wykresy...")
    fig = plt.figure(figsize=(20, 16), facecolor="#0a1520")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32, left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_map  = fig.add_subplot(gs[0, :])
    ax_deg  = fig.add_subplot(gs[1, 0])
    ax_heat = fig.add_subplot(gs[1, 1])

    plot_network_map(G, page_rank, ax_map)
    plot_degree_distribution(G, ax_deg)
    plot_simulation_heatmap(sim_stats, ax_heat)

    fig.suptitle("SYMULACJA SIECI ZŁOŻONEJ LOTNISK W POLSCE (Z ZAKŁÓCENIAMI POGODOWYMI)", fontsize=15, fontweight="bold", color="white", y=0.97)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Zapisano: {output_path}")

# =============================================================================
# MAIN PUNKT WEJŚCIA
# =============================================================================
def main():
    # 1. Uruchomienie symulacji SimPy (Tekstowy raport kryzysowy w konsoli)
    run_simulation()

    # 2. Budowa grafu i wyliczenie PageRank na potrzeby mapy sieci
    G = build_graph()
    page_rank = nx.pagerank(G, weight="freq_daily")

    # 3. Wygenerowanie wykresów z rzeczywistych danych zapisanych w `stats`
    generate_visualizations(G, page_rank, stats, output_path="wizualizacja_pogoda.png")

if __name__ == "__main__":
    main()