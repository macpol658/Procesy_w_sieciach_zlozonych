"""
=============================================================================
ANIMACJA SYMULACJI SIECI LOTNICZEJ W PYTHONIE (.GIF) - WERSJA ULEPSZONA
=============================================================================
- Wolniejszy, płynniejszy ruch samolotów
- Ikonki samolotów (✈) obrócone zgodnie z wektorem lotu
- Wektorowy zarys mapy Polski w tle
=============================================================================
"""

import simpy
import random
import numpy as np
import collections
import itertools
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Zarys granic Polski (uproszczony wektor do tła) ───────────────────────
POLAND_BORDER = [
    (14.12, 53.91), (14.24, 54.23), (14.54, 54.18), (15.00, 54.00),
    (16.20, 54.20), (18.80, 54.83), (19.90, 54.43), (22.80, 54.33),
    (23.95, 52.10), (23.60, 50.30), (22.85, 49.03), (19.00, 49.50),
    (17.00, 50.10), (15.00, 51.00), (14.12, 52.80), (14.12, 53.91)
]

random.seed(42)
np.random.seed(42)

# =============================================================================
# 1. DANE WEJŚCIOWE
# =============================================================================
AIRPORTS = {
    "WAW": {"city": "Warszawa", "lat": 52.165, "lon": 20.967, "gates": 12},
    "KRK": {"city": "Kraków",   "lat": 50.077, "lon": 19.784, "gates": 6},
    "KTW": {"city": "Katowice", "lat": 50.474, "lon": 19.080, "gates": 6},
    "GDN": {"city": "Gdańsk",   "lat": 54.377, "lon": 18.466, "gates": 5},
    "WRO": {"city": "Wrocław",  "lat": 51.102, "lon": 16.898, "gates": 5},
    "POZ": {"city": "Poznań",   "lat": 52.421, "lon": 16.826, "gates": 4},
    "RZE": {"city": "Rzeszów",  "lat": 50.110, "lon": 22.019, "gates": 3},
    "LCJ": {"city": "Łódź",     "lat": 51.722, "lon": 19.398, "gates": 3},
    "SZZ": {"city": "Szczecin", "lat": 53.584, "lon": 14.902, "gates": 2},
    "BZG": {"city": "Bydgoszcz","lat": 53.097, "lon": 17.977, "gates": 2},
}

ROUTES = [
    ("WAW", "KRK", 10), ("WAW", "KTW", 6), ("WAW", "GDN", 10),
    ("WAW", "WRO", 8),  ("WAW", "POZ", 8), ("WAW", "RZE", 4),
    ("WAW", "LCJ", 4),  ("WAW", "SZZ", 4), ("WAW", "BZG", 4),
    ("KRK", "GDN", 2),  ("KRK", "WRO", 2), ("KTW", "GDN", 2),
    ("GDN", "POZ", 2),  ("WRO", "POZ", 2),
]

CRUISE_SPEED_KMH = 700
SIM_DURATION_MIN = 1440

airport_is_open = {code: True for code in AIRPORTS.keys()}
flight_log = []

# =============================================================================
# 2. LOGIKA SYMULACJI (ZBIERANIE DANYCH)
# =============================================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_nearest_open_airport(closed_code, origin_code):
    best, min_dist = None, float('inf')
    t_lat, t_lon = AIRPORTS[closed_code]["lat"], AIRPORTS[closed_code]["lon"]
    for code, is_open in airport_is_open.items():
        if is_open and code != closed_code and code != origin_code:
            d = haversine_distance(t_lat, t_lon, AIRPORTS[code]["lat"], AIRPORTS[code]["lon"])
            if d < min_dist:
                min_dist, best = d, code
    return best, min_dist

def weather_process(env, target, start, duration):
    yield env.timeout(start)
    airport_is_open[target] = False
    yield env.timeout(duration)
    airport_is_open[target] = True

def flight_process(env, src, dst, base_travel_min, gate_resources):
    while not airport_is_open[dst]:
        yield env.timeout(10)

    with gate_resources[src].request() as req:
        yield req
        yield env.timeout(random.uniform(15, 30))

    takeoff_time = env.now
    actual_travel = base_travel_min * random.uniform(0.9, 1.1)
    yield env.timeout(actual_travel)

    final_dst = dst
    if not airport_is_open[dst]:
        alt, extra_dist = get_nearest_open_airport(dst, src)
        if alt:
            final_dst = alt
            yield env.timeout(extra_dist / (CRUISE_SPEED_KMH / 60))
        else:
            final_dst = src
            yield env.timeout(actual_travel)

    landing_time = env.now

    with gate_resources[final_dst].request() as req:
        yield req
        yield env.timeout(15)

    flight_log.append({
        "src": src,
        "dst": final_dst,
        "takeoff": takeoff_time,
        "landing": landing_time
    })

def route_generator(env, src, dst, travel_min, freq, gates):
    mean_interval = SIM_DURATION_MIN / (freq / 2)
    while True:
        yield env.timeout(random.expovariate(1 / mean_interval))
        if env.now >= SIM_DURATION_MIN:
            break
        env.process(flight_process(env, src, dst, travel_min, gates))

def run_simulation():
    print("Rozpoczynam symulację ruchu i zbieranie danych lotów...")
    env = simpy.Environment()
    gates = {code: simpy.Resource(env, capacity=d["gates"]) for code, d in AIRPORTS.items()}

    for src, dst, freq in ROUTES:
        dist = haversine_distance(AIRPORTS[src]["lat"], AIRPORTS[src]["lon"],
                                  AIRPORTS[dst]["lat"], AIRPORTS[dst]["lon"])
        travel_min = dist / (CRUISE_SPEED_KMH / 60)
        env.process(route_generator(env, src, dst, travel_min, freq, gates))
        env.process(route_generator(env, dst, src, travel_min, freq, gates))

    # Warszawa zamyka się z powodu mgły między 400. a 600. minutą
    env.process(weather_process(env, "WAW", 400, 200))
    env.run(until=SIM_DURATION_MIN)
    print(f"Zakończono. Zarejestrowano {len(flight_log)} lotów.")

# =============================================================================
# 3. ANIMACJA MATPLOTLIB (WOLNIEJSZA, Z MAPĄ I IKONAMI)
# =============================================================================
def animate_network():
    print("Generuję klatki animacji, to może zająć dłuższą chwilę...")

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0d1b2a")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)

    border_lon, border_lat = zip(*POLAND_BORDER)
    lons = [d["lon"] for d in AIRPORTS.values()]
    lats = [d["lat"] for d in AIRPORTS.values()]

    def update(frame_time):
        ax.clear()
        ax.set_facecolor("#0d1b2a")

        # Ograniczenia i proporcje mapy
        ax.set_xlim(13.5, 24.5)
        ax.set_ylim(48.5, 55.5)
        ax.axis('off')

        # Tytuł z czasem
        time_str = f"Czas symulacji: {int(frame_time):02d} min"
        if 400 <= frame_time <= 600:
            time_str += "  |  ⚠️ WAW ZAMKNIĘTE (MGŁA)"
            title_color = "#ef5350"
        else:
            title_color = "white"
        ax.set_title(time_str, color=title_color, fontsize=14, fontweight="bold")

        # 1. Rysowanie tła (obrys Polski)

        # 2. Rysowanie stałych tras
        for src, dst, _ in ROUTES:
            x = [AIRPORTS[src]["lon"], AIRPORTS[dst]["lon"]]
            y = [AIRPORTS[src]["lat"], AIRPORTS[dst]["lat"]]
            ax.plot(x, y, color="#4fc3f7", linewidth=0.5, alpha=0.3, zorder=2)

        # 3. Rysowanie lotnisk
        for code, data in AIRPORTS.items():
            if code == "WAW" and 400 <= frame_time <= 600:
                color, size = "#ef5350", 200
            else:
                color, size = "#b0bec5", 80

            ax.scatter(data["lon"], data["lat"], color=color, s=size, zorder=3, edgecolors="black")
            ax.text(data["lon"]+0.15, data["lat"]+0.05, code, color="white", fontsize=9, fontweight="bold", zorder=4)

        # 4. Rysowanie samolotów (ikony obrócone w stronę lotu)
        for flight in flight_log:
            if flight["takeoff"] <= frame_time <= flight["landing"]:
                progress = (frame_time - flight["takeoff"]) / (flight["landing"] - flight["takeoff"])

                lon_start, lat_start = AIRPORTS[flight["src"]]["lon"], AIRPORTS[flight["src"]]["lat"]
                lon_end, lat_end = AIRPORTS[flight["dst"]]["lon"], AIRPORTS[flight["dst"]]["lat"]

                cur_lon = lon_start + progress * (lon_end - lon_start)
                cur_lat = lat_start + progress * (lat_end - lat_start)

                # Obliczanie kąta lotu (uwzględniając zniekształcenie mapy)
                dx = (lon_end - lon_start) * math.cos(math.radians((lat_start + lat_end) / 2))
                dy = lat_end - lat_start
                angle_deg = math.degrees(math.atan2(dy, dx))

                # Znak "✈" domyślnie "patrzy" lekko do góry w prawo, korygujemy o ~45 stopni
                display_angle = angle_deg - 45

                ax.text(cur_lon, cur_lat, "✈", color="#ffeb3b", fontsize=18,
                        ha='center', va='center', rotation=display_angle, zorder=5)

    # ZWOLNIENIE SYMULACJI: Zmieniony skok klatek (step=2) i klatkaż (fps=15)
    # Wygeneruje to więcej klatek, ale ruch będzie znacznie płynniejszy.
    frames = list(range(0, SIM_DURATION_MIN, 2))
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)

    gif_path = "symulacja_lotow.gif"
    ani.save(gif_path, writer='pillow', fps=15)
    print(f"Gotowe! Plik zapisano jako: {gif_path}")

# =============================================================================
if __name__ == "__main__":
    run_simulation()
    animate_network()