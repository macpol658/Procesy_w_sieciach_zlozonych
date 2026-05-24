"""
=============================================================================
SYMULACJA SIECI ZŁOŻONEJ LOTNISK W POLSCE
=============================================================================
Źródło danych: ULC, "Szacowane przepustowości portów lotniczych", lipiec 2009
Narzędzia: SimPy (symulacja), NetworkX (analiza sieci), igraph (graph analytics)
=============================================================================

CO SYMULUJEMY?
  Każde lotnisko to węzeł sieci z:
    - przepustowością (max pas./rok wg ULC 2009)
    - rzeczywistym wykorzystaniem (% w roku 2009)
    - zasobem stanowisk odprawy (SimPy Resource)

  Każde połączenie to krawędź z:
    - odległością (km)
    - liczbą operacji/dobę (freq)
    - czasem lotu (min)

  Symulowany RUCH:
    - loty generowane są losowo wg rozkładu Poissona dla każdej trasy
    - każdy lot zajmuje stanowiska odprawy na obu lotniskach
    - mierzymy: opóźnienia, kolejki, obciążenie węzłów
"""

import simpy
import networkx as nx
import igraph as ig
import random
import numpy as np
import collections
import itertools

# ── deterministyczny ziarno dla powtarzalności ──────────────────────────────
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DANE WEJŚCIOWE – Polskie porty lotnicze (ULC 2009)
# =============================================================================
# Przepustowość w tys. pasażerów / rok (szacowane), wykorzystanie ~rzeczywiste
# Dane z tabeli ULC (lipiec 2009) + oficjalne statystyki ULC za 2009 r.

AIRPORTS = {
    "WAW": {
        "name": "Warszawa Okęcie",
        "city": "Warszawa",
        "capacity_mpax": 9.5,       # mln pas./rok (przepustowość)
        "actual_mpax": 7.9,         # mln pas./rok (ruch rzeczywisty 2009)
        "gates": 12,                # liczba stanowisk (→ SimPy Resource)
        "lat": 52.165, "lon": 20.967,
    },
    "KRK": {
        "name": "Kraków J. Paweł II",
        "city": "Kraków",
        "capacity_mpax": 5.0,
        "actual_mpax": 2.8,
        "gates": 6,
        "lat": 50.077, "lon": 19.784,
    },
    "KTW": {
        "name": "Katowice-Pyrzowice",
        "city": "Katowice",
        "capacity_mpax": 4.5,
        "actual_mpax": 2.6,
        "gates": 6,
        "lat": 50.474, "lon": 19.080,
    },
    "GDN": {
        "name": "Gdańsk im. Wałęsy",
        "city": "Gdańsk",
        "capacity_mpax": 3.5,
        "actual_mpax": 2.0,
        "gates": 5,
        "lat": 54.377, "lon": 18.466,
    },
    "WRO": {
        "name": "Wrocław Strachowice",
        "city": "Wrocław",
        "capacity_mpax": 3.0,
        "actual_mpax": 1.7,
        "gates": 5,
        "lat": 51.102, "lon": 16.898,
    },
    "POZ": {
        "name": "Poznań-Ławica",
        "city": "Poznań",
        "capacity_mpax": 2.5,
        "actual_mpax": 1.4,
        "gates": 4,
        "lat": 52.421, "lon": 16.826,
    },
    "RZE": {
        "name": "Rzeszów-Jasionka",
        "city": "Rzeszów",
        "capacity_mpax": 1.2,
        "actual_mpax": 0.5,
        "gates": 3,
        "lat": 50.110, "lon": 22.019,
    },
    "LCJ": {
        "name": "Łódź Władysława Reymonta",
        "city": "Łódź",
        "capacity_mpax": 1.5,
        "actual_mpax": 0.4,
        "gates": 3,
        "lat": 51.722, "lon": 19.398,
    },
    "SZZ": {
        "name": "Szczecin-Goleniów",
        "city": "Szczecin",
        "capacity_mpax": 0.8,
        "actual_mpax": 0.3,
        "gates": 2,
        "lat": 53.584, "lon": 14.902,
    },
    "BZG": {
        "name": "Bydgoszcz im. Ignacego Jana Paderewskiego",
        "city": "Bydgoszcz",
        "capacity_mpax": 0.8,
        "actual_mpax": 0.3,
        "gates": 2,
        "lat": 53.097, "lon": 17.977,
    },
}

# Połączenia krajowe (trasy wewnętrzne) — lista (skąd, dokąd)
# Trasy dwukierunkowe; freq = średnia liczba operacji (2 kierunki) na dobę
ROUTES = [
    # Hub WAW z regionalnymi
    ("WAW", "KRK", {"dist_km": 252, "freq_daily": 10}),
    ("WAW", "KTW", {"dist_km": 295, "freq_daily": 6}),
    ("WAW", "GDN", {"dist_km": 340, "freq_daily": 10}),
    ("WAW", "WRO", {"dist_km": 347, "freq_daily": 8}),
    ("WAW", "POZ", {"dist_km": 310, "freq_daily": 8}),
    ("WAW", "RZE", {"dist_km": 295, "freq_daily": 4}),
    ("WAW", "LCJ", {"dist_km": 121, "freq_daily": 4}),
    ("WAW", "SZZ", {"dist_km": 520, "freq_daily": 4}),
    ("WAW", "BZG", {"dist_km": 360, "freq_daily": 4}),
    # Trasy regionalne (cross-connections)
    ("KRK", "GDN", {"dist_km": 550, "freq_daily": 2}),
    ("KRK", "WRO", {"dist_km": 267, "freq_daily": 2}),
    ("KTW", "GDN", {"dist_km": 560, "freq_daily": 2}),
    ("GDN", "POZ", {"dist_km": 336, "freq_daily": 2}),
    ("WRO", "POZ", {"dist_km": 165, "freq_daily": 2}),
]

# Prędkość przelotowa (km/h) — typowa dla krótkiego dystansu
CRUISE_SPEED_KMH = 700
# Symulacja: 1 jednostka czasu = 1 minuta; symulujemy 1 dobę (1440 min)
SIM_DURATION_MIN = 1440
# Średnia liczba pasażerów na lot
PAX_PER_FLIGHT = 150


# =============================================================================
# 2. BUDOWA GRAFU – NetworkX
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
        # Dodajemy w obu kierunkach (ruch dwustronny)
        for a, b in [(src, dst), (dst, src)]:
            G.add_edge(
                a, b,
                dist_km=attrs["dist_km"],
                freq_daily=attrs["freq_daily"],
                travel_min=travel_min,
                # waga krawędzi: odwrotność częstotliwości (im więcej lotów, tym "bliżej")
                weight=1.0 / attrs["freq_daily"],
            )

    return G


# =============================================================================
# 3. ANALIZA SIECI – NetworkX (metryki topologiczne)
# =============================================================================

def analyze_networkx(G: nx.DiGraph):
    print("\n" + "="*70)
    print("  ANALIZA TOPOLOGICZNA SIECI – NetworkX")
    print("="*70)

    print(f"\n  Węzły (lotniska)    : {G.number_of_nodes()}")
    print(f"  Krawędzie (połącz.) : {G.number_of_edges()}")
    print(f"  Gęstość grafu       : {nx.density(G):.4f}")

    # --- Centralność --------------------------------------------------------
    degree_c   = nx.degree_centrality(G)
    between_c  = nx.betweenness_centrality(G, normalized=True, weight="weight")
    closeness_c = nx.closeness_centrality(G, distance="weight")
    page_rank  = nx.pagerank(G, weight="freq_daily")   # oparty na ruchu

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

    # --- Przepustowość węzłów -----------------------------------------------
    print("\n  ┌────────────────────────────────────────────────────────────────┐")
    print("  │ Przepustowość i wykorzystanie portów (dane ULC 2009)           │")
    print("  ├──────┬──────────────────────────┬──────────┬─────────┬────────┤")
    print("  │ IATA │ Port                     │Przep.(M) │Ruch.(M) │Wykor.% │")
    print("  ├──────┼──────────────────────────┼──────────┼─────────┼────────┤")
    for code in sorted_airports:
        a = AIRPORTS[code]
        bar = "█" * int(a["actual_mpax"] / a["capacity_mpax"] * 10)
        util_pct = a["actual_mpax"] / a["capacity_mpax"] * 100
        name = a["name"][:26]
        print(
            f"  │ {code} │ {name:<26} │   {a['capacity_mpax']:5.1f}  │"
            f"  {a['actual_mpax']:5.1f}  │ {util_pct:5.1f}% │"
        )
    print("  └──────┴──────────────────────────┴──────────┴─────────┴────────┘")

    # --- Ścieżki i spójność -------------------------------------------------
    G_undir = G.to_undirected()
    print(f"\n  Spójność grafu         : {'TAK' if nx.is_connected(G_undir) else 'NIE'}")
    try:
        avg_path = nx.average_shortest_path_length(G_undir, weight="weight")
        print(f"  Śr. długość ścieżki    : {avg_path:.3f} (wg wagi 1/freq)")
    except Exception:
        print("  Śr. długość ścieżki    : (graf niespójny)")

    # Najkrótsza trasa WAW → RZE (wg liczby przesiadek)
    path = nx.shortest_path(G, "WAW", "RZE", weight=None)
    print(f"\n  Najkrótsza trasa WAW → RZE (min. przesiadek): {' → '.join(path)}")

    # Max-flow (przepustowość ruchu) WAW → KRK
    flow_val, _ = nx.maximum_flow(G, "WAW", "KRK", capacity="freq_daily")
    print(f"  Max-flow WAW → KRK     : {flow_val:.0f} lotów/dobę")

    return degree_c, between_c, page_rank


# =============================================================================
# 4. ANALIZA SIECI – igraph (społeczności, clustering)
# =============================================================================

def analyze_igraph(G_nx: nx.DiGraph):
    print("\n" + "="*70)
    print("  ANALIZA GRAFU – igraph (community detection, clustering)")
    print("="*70)

    # Konwersja NetworkX → igraph
    nodes = list(G_nx.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    edges_ig = [(node_idx[u], node_idx[v]) for u, v in G_nx.edges()]
    weights_ig = [G_nx[u][v]["freq_daily"] for u, v in G_nx.edges()]

    g = ig.Graph(n=len(nodes), edges=edges_ig, directed=True)
    g.vs["name"] = nodes
    g.vs["label"] = [AIRPORTS[n]["city"] for n in nodes]
    g.es["weight"] = weights_ig

    # Współczynnik klasteryzacji (undirected)
    g_undir = g.as_undirected(combine_edges="sum")

    print("\n  Współczynnik klasteryzacji (lokalny) – igraph:")
    clustering = g_undir.transitivity_local_undirected(mode="zero")
    for i, v in enumerate(g_undir.vs):
        print(f"    {nodes[i]} ({AIRPORTS[nodes[i]]['city']:<15}): {clustering[i]:.4f}")

    global_clust = g_undir.transitivity_undirected()
    print(f"\n  Globalny wsp. klasteryzacji: {global_clust:.4f}")

    # PageRank (igraph)
    pr_ig = g.pagerank(weights="weight")
    print("\n  PageRank (igraph, wg freq_daily):")
    pr_sorted = sorted(zip(nodes, pr_ig), key=lambda x: -x[1])
    for code, pr in pr_sorted:
        bar = "█" * int(pr * 200)
        print(f"    {code}: {pr:.5f}  {bar}")

    # Community detection – Louvain na nieskierowanym
    try:
        communities = g_undir.community_multilevel(weights="weight")
        print(f"\n  Detekcja społeczności (Louvain): {len(communities)} grupy/grup")
        for i, comm in enumerate(communities):
            members = [nodes[j] for j in comm]
            print(f"    Grupa {i+1}: {members}")
        print(f"  Modularność: {communities.modularity:.4f}")
    except Exception as e:
        print(f"  Community detection: {e}")

    # Diameter (średnica grafu)
    try:
        diam = g_undir.diameter(weights=None)
        print(f"\n  Średnica grafu (hop distance): {diam}")
    except Exception:
        pass

    return g


# =============================================================================
# 5. SYMULACJA RUCHU – SimPy
# =============================================================================
"""
Model symulacji:

  GENEROWANIE LOTÓW:
    Dla każdej trasy (A→B) loty są generowane w procesie Poissona
    z intensywnością λ = freq_daily / (24*60) [lotów/min].

  KAŻDY LOT (proces SimPy):
    1. Żąda `gate_slots` na lotnisku A (departure gate)   → może czekać w kolejce
    2. Trwa odprawianie + boarding (normal. ~20–40 min)
    3. Zwalnia gate na A, leci (timeout = travel_min + losowe odchylenie ±10%)
    4. Żąda `gate_slots` na lotnisku B (arrival gate)     → może czekać w kolejce
    5. Trwa deboarding + handling (~15–30 min)
    6. Zwalnia gate na B

  ZBIERANE STATYSTYKI:
    - Łączne loty dla każdej trasy
    - Pasażerowie obsłużeni przez każde lotnisko
    - Czas oczekiwania w kolejce na gate (opóźnienie)
    - Aktualne wykorzystanie gatów (utilization)
"""

# Statystyki globalne
stats = {
    "flights_completed": collections.Counter(),
    "flights_delayed": collections.Counter(),
    "total_delay_min": collections.Counter(),
    "pax_handled": collections.Counter(),
    "gate_wait_times": collections.defaultdict(list),
}


def flight_process(env, flight_id, src, dst, travel_min, gate_resources):
    """Pojedynczy lot jako proces SimPy."""
    pax = random.randint(int(PAX_PER_FLIGHT * 0.7), int(PAX_PER_FLIGHT * 1.1))

    # --- DEPARTURE ---
    departure_request_time = env.now
    with gate_resources[src].request() as req:
        yield req                                       # czekamy na wolny gate
        gate_wait = env.now - departure_request_time

        if gate_wait > 0:
            stats["flights_delayed"][src] += 1
            stats["total_delay_min"][src] += gate_wait
        stats["gate_wait_times"][src].append(gate_wait)

        # Obsługa odlotu: check-in + boarding
        boarding_time = random.normalvariate(25, 5)
        yield env.timeout(max(boarding_time, 10))

    # Pasażerowie opuszczają lotnisko A
    stats["pax_handled"][src] += pax

    # --- LOT ---
    actual_travel = travel_min * random.uniform(0.9, 1.1)   # ±10% zmienność
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

        # Deboarding + handling
        deboard_time = random.normalvariate(20, 4)
        yield env.timeout(max(deboard_time, 10))

    stats["pax_handled"][dst] += pax
    stats["flights_completed"][f"{src}→{dst}"] += 1


def route_generator(env, src, dst, travel_min, freq_daily, gate_resources):
    """Generator lotów na danej trasie (rozkład Poissona)."""
    # Średni czas między lotami (min)
    mean_interarrival = SIM_DURATION_MIN / (freq_daily / 2)   # /2 bo jeden kierunek
    flight_counter = itertools.count(1)

    while True:
        # Losowy odstęp (exp. → rozkład Poissona)
        interarrival = random.expovariate(1 / mean_interarrival)
        yield env.timeout(interarrival)
        if env.now >= SIM_DURATION_MIN:
            break
        fid = f"{src}{dst}_{next(flight_counter):04d}"
        env.process(flight_process(env, fid, src, dst, travel_min, gate_resources))


def run_simulation(G_nx: nx.DiGraph):
    print("\n" + "="*70)
    print("  SYMULACJA RUCHU LOTNICZEGO – SimPy (1 doba = 1440 min)")
    print("="*70)

    env = simpy.Environment()

    # Zasoby: gate_resources[airport] = SimPy Resource z capacity=gates
    gate_resources = {
        code: simpy.Resource(env, capacity=data["gates"])
        for code, data in AIRPORTS.items()
    }

    # Uruchom generator lotów dla każdej trasy w obu kierunkach
    for src, dst, attrs in ROUTES:
        travel_min = round(attrs["dist_km"] / CRUISE_SPEED_KMH * 60)
        for a, b in [(src, dst), (dst, src)]:
            env.process(route_generator(env, a, b, travel_min,
                                        attrs["freq_daily"], gate_resources))

    env.run(until=SIM_DURATION_MIN)

    # ------- WYNIKI SYMULACJI -----------------------------------------------
    print(f"\n  Symulacja zakończona. Czas: {SIM_DURATION_MIN} min (1 doba)")

    # Podsumowanie tras
    print("\n  ── Ukończone loty wg tras ──────────────────────────────────────")
    total_flights = 0
    for route, count in sorted(stats["flights_completed"].items(), key=lambda x: -x[1]):
        total_flights += count
        print(f"    {route}: {count:3d} lotów")
    print(f"    ───────────────────────────────")
    print(f"    RAZEM: {total_flights} lotów")

    # Opóźnienia i obciążenie węzłów
    print("\n  ── Statystyki lotnisk (wyniki symulacji) ───────────────────────")
    print(f"  {'IATA':<6} {'Miasto':<15} {'Pax':<8} {'Op.opóźn.':<12} {'Śr.opóźn.':<12} {'Wykor.%'}")
    print("  " + "─"*66)

    for code, data in AIRPORTS.items():
        pax = stats["pax_handled"].get(code, 0)
        delayed = stats["flights_delayed"].get(code, 0)
        total_delay = stats["total_delay_min"].get(code, 0)
        wait_times = stats["gate_wait_times"].get(code, [0])
        avg_wait = np.mean(wait_times) if wait_times else 0.0

        # Symulowane wykorzystanie: pax vs. przepustowość 1-dniowa
        daily_cap = data["capacity_mpax"] * 1_000_000 / 365
        sim_util = min(pax / daily_cap * 100, 100) if daily_cap > 0 else 0

        print(
            f"  {code:<6} {data['city']:<15} {pax:>7,}  "
            f"{delayed:>10}  {avg_wait:>9.1f} min  {sim_util:>6.1f}%"
        )

    return stats, gate_resources


# =============================================================================
# 6. SYNTETYCZNE METRYKI SIECI ZŁOŻONEJ
# =============================================================================

def complex_network_metrics(G_nx: nx.DiGraph, between_c: dict):
    print("\n" + "="*70)
    print("  METRYKI SIECI ZŁOŻONEJ")
    print("="*70)

    G_undir = G_nx.to_undirected()

    # Rozkład stopni
    degrees = dict(G_nx.degree())
    deg_values = list(degrees.values())
    print(f"\n  Rozkład stopni (degree distribution):")
    print(f"    Min stopień  : {min(deg_values)}")
    print(f"    Max stopień  : {max(deg_values)}")
    print(f"    Śr. stopień  : {np.mean(deg_values):.2f}")
    print(f"    Odch. std.   : {np.std(deg_values):.2f}")

    # Power-law check (sieć bezskalowa?)
    # Prosty test: czy wariancja >> średnia^2 (heavy tail)?
    cv = np.std(deg_values) / np.mean(deg_values)
    print(f"    Wsp. zmienności: {cv:.3f} ", end="")
    if cv > 0.5:
        print("→ topologia HUB-and-SPOKE (WAW dominuje jako hub)")
    else:
        print("→ sieć bardziej jednorodna")

    # Węzeł krytyczny (bottleneck – najwyższa pośredniość)
    bottleneck = max(between_c, key=between_c.get)
    print(f"\n  Węzeł krytyczny (max betweenness): {bottleneck} "
          f"({AIRPORTS[bottleneck]['city']}) = {between_c[bottleneck]:.4f}")
    print(f"  → Awaria {bottleneck} najbardziej zakłóci sieć krajową")

    # Odporność (robustness) – ile węzłów usunąć, by sieć rozspójnić?
    print(f"\n  Test odporności (usuwanie węzłów wg betweenness):")
    G_test = G_undir.copy()
    removed = []
    sorted_nodes = sorted(between_c, key=between_c.get, reverse=True)
    for node in sorted_nodes:
        if not nx.is_connected(G_test):
            break
        G_test.remove_node(node)
        removed.append(node)
        connected = nx.is_connected(G_test)
        print(f"    Usunięto {node}: sieć {'spójna ✓' if connected else 'NIESPÓJNA ✗'}")
        if not connected:
            break

    # Small-world?
    try:
        avg_l = nx.average_shortest_path_length(G_undir, weight=None)
        # Porównanie z losowym grafem Erdos-Renyi o tych samych parametrach
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
# 7. MAIN
# =============================================================================

def main():
    print("\n" + "█"*70)
    print("  SIEĆ ZŁOŻONA POLSKICH PORTÓW LOTNICZYCH – 2009")
    print("  SimPy + NetworkX + igraph")
    print("█"*70)

    # 2. Graf NetworkX
    G = build_networkx_graph()

    # 3. Analiza NetworkX
    deg_c, bet_c, pr = analyze_networkx(G)

    # 4. Analiza igraph
    g_ig = analyze_igraph(G)

    # 5. Symulacja SimPy
    sim_stats, gate_res = run_simulation(G)

    # 6. Metryki sieci złożonej
    complex_network_metrics(G, bet_c)

    print("\n" + "█"*70)
    print("  SYMULACJA ZAKOŃCZONA")
    print("█"*70 + "\n")

    return G, g_ig, sim_stats


if __name__ == "__main__":
    G, g_ig, sim_stats = main()