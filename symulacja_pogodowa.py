"""
=============================================================================
SYMULACJA SIECI ZŁOŻONEJ LOTNISK W POLSCE (Z MODUŁEM POGODOWYM)
=============================================================================
Źródło danych: ULC, "Szacowane przepustowości portów lotniczych", lipiec 2009
Nowe funkcje:
 - Zdarzenia losowe (np. gęsta mgła zamykająca lotnisko)
 - Ground Delay Program (wstrzymywanie startów)
 - Diversion (przekierowania lotów w powietrzu do najbliższego geograficznie)
=============================================================================
"""

import simpy
import networkx as nx
import igraph as ig
import random
import numpy as np
import collections
import itertools
import math

# ── deterministyczny ziarno dla powtarzalności ──────────────────────────────
random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DANE WEJŚCIOWE I ZMIENNE GLOBALNE
# =============================================================================

AIRPORTS = {
    "WAW": {"name": "Warszawa Okęcie", "city": "Warszawa", "capacity_mpax": 9.5, "actual_mpax": 7.9, "gates": 12,
            "lat": 52.165, "lon": 20.967},
    "KRK": {"name": "Kraków J. Paweł II", "city": "Kraków", "capacity_mpax": 5.0, "actual_mpax": 2.8, "gates": 6,
            "lat": 50.077, "lon": 19.784},
    "KTW": {"name": "Katowice-Pyrzowice", "city": "Katowice", "capacity_mpax": 4.5, "actual_mpax": 2.6, "gates": 6,
            "lat": 50.474, "lon": 19.080},
    "GDN": {"name": "Gdańsk im. Wałęsy", "city": "Gdańsk", "capacity_mpax": 3.5, "actual_mpax": 2.0, "gates": 5,
            "lat": 54.377, "lon": 18.466},
    "WRO": {"name": "Wrocław Strachowice", "city": "Wrocław", "capacity_mpax": 3.0, "actual_mpax": 1.7, "gates": 5,
            "lat": 51.102, "lon": 16.898},
    "POZ": {"name": "Poznań-Ławica", "city": "Poznań", "capacity_mpax": 2.5, "actual_mpax": 1.4, "gates": 4,
            "lat": 52.421, "lon": 16.826},
    "RZE": {"name": "Rzeszów-Jasionka", "city": "Rzeszów", "capacity_mpax": 1.2, "actual_mpax": 0.5, "gates": 3,
            "lat": 50.110, "lon": 22.019},
    "LCJ": {"name": "Łódź W. Reymonta", "city": "Łódź", "capacity_mpax": 1.5, "actual_mpax": 0.4, "gates": 3,
            "lat": 51.722, "lon": 19.398},
    "SZZ": {"name": "Szczecin-Goleniów", "city": "Szczecin", "capacity_mpax": 0.8, "actual_mpax": 0.3, "gates": 2,
            "lat": 53.584, "lon": 14.902},
    "BZG": {"name": "Bydgoszcz Paderewskiego", "city": "Bydgoszcz", "capacity_mpax": 0.8, "actual_mpax": 0.3,
            "gates": 2, "lat": 53.097, "lon": 17.977},
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

# --- ZMIENNE GLOBALNE DLA NOWEJ LOGIKI ---
airport_is_open = {code: True for code in AIRPORTS.keys()}

stats = {
    "flights_completed": collections.Counter(),
    "flights_delayed": collections.Counter(),
    "total_delay_min": collections.Counter(),
    "pax_handled": collections.Counter(),
    "gate_wait_times": collections.defaultdict(list),
    "diverted_flights": collections.Counter(),  # NOWE
    "weather_delays_min": collections.Counter(),  # NOWE
}


# =============================================================================
# 2. LOGIKA GEOGRAFICZNA (HAVERSINE)
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Oblicza odległość w linii prostej (km) między dwiema współrzędnymi."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
        dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def get_nearest_open_airport(closed_airport_code, origin_airport_code):
    """Znajduje najbliższe OTWARTE lotnisko względem tego zamkniętego."""
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
    """Zdarzenie losowe: Zamknięcie lotniska z powodu pogody."""
    yield env.timeout(start_time_min)
    print(f"\n[Czas: {env.now:>4.0f} min] ⚠️ ALARM POGODOWY: Lotnisko {target_airport} zostaje ZAMKNIĘTE (np. mgła)!")
    airport_is_open[target_airport] = False

    yield env.timeout(duration_min)
    print(f"\n[Czas: {env.now:>4.0f} min] ☀️ POPRAWA POGODY: Lotnisko {target_airport} ponownie OTWARTE.")
    airport_is_open[target_airport] = True


def flight_process(env, flight_id, src, dst, travel_min, gate_resources):
    """Model pojedynczego lotu."""
    pax = random.randint(int(PAX_PER_FLIGHT * 0.7), int(PAX_PER_FLIGHT * 1.1))

    # --- DEPARTURE ---
    departure_request_time = env.now

    # 1. GROUND DELAY (Czekamy, aż cel się otworzy zanim zajmiemy gate i polecimy)
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

    # 2. DIVERSION (Lotnisko zamknięto gdy byliśmy w powietrzu)
    if not airport_is_open[dst]:
        nearest_airport, extra_dist_km = get_nearest_open_airport(dst, src)

        if nearest_airport:
            dst = nearest_airport
            extra_travel_min = extra_dist_km / (CRUISE_SPEED_KMH / 60)
            print(f"[Czas: {env.now:>4.0f} min] 🔀 PRZEKIEROWANIE: Lot {flight_id} ({src}→{original_dst}). "
                  f"Cel ZAMKNIĘTY. Leci do: {dst} (+{extra_dist_km:.0f} km / +{extra_travel_min:.0f} min).")

            stats["diverted_flights"][original_dst] += 1
            yield env.timeout(extra_travel_min)
        else:
            print(
                f"[Czas: {env.now:>4.0f} min] ⚠️ KRYZYS: Lot {flight_id} nie ma otwartej alternatywy! Wraca do {src}.")
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
    print("  SYMULACJA RUCHU LOTNICZEGO – SimPy (z analizą zakłóceń)")
    print("=" * 70)

    env = simpy.Environment()
    gate_resources = {code: simpy.Resource(env, capacity=data["gates"]) for code, data in AIRPORTS.items()}

    for src, dst, attrs in ROUTES:
        travel_min = round(attrs["dist_km"] / CRUISE_SPEED_KMH * 60)
        for a, b in [(src, dst), (dst, src)]:
            env.process(route_generator(env, a, b, travel_min, attrs["freq_daily"], gate_resources))

    # --- ZDARZENIE LOSOWE ---
    # Symulujemy zamknięcie Lotniska Chopina (WAW) we mgle na 3 godziny (od 12:00 do 15:00 w symulacji)
    env.process(weather_disruption_process(env, target_airport="WAW", start_time_min=720, duration_min=180))

    env.run(until=SIM_DURATION_MIN)

    # --- WYNIKI ---
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

        print(
            f"  {code:<5} {data['city']:<12} {pax:>6,}  {delayed:>7}  {avg_wait:>7.1f}m  {diverts:>6}  {weather_wait:>12}m")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    run_simulation()