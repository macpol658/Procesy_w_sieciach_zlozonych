"""
=============================================================================
SYMULACJA KRYZYSOWA (POPRAWIONE ŚLEDZENIE KOLEJKI LIVE W WĘZŁACH)
=============================================================================
"""

import simpy
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# KONFIGURACJA GŁÓWNA
# =============================================================================
SIM_DURATION_MIN = 1440
TRAFFIC_MULTIPLIER = 3


ANOMALIES_CONFIG = [
    ("KRK", 550, 240),
    ("LCJ", 200, 150),
    ("POZ", 700, 60),
    ("LUZ", 300, 120),
    ("WAW", 300, 240)
]


CRUISE_SPEED_KMH = 700
random.seed(42)

AIRPORTS = {
    "WAW": {"name": "Warszawa Okęcie", "gates": 12, "lat": 52.165, "lon": 20.967},
    "KRK": {"name": "Kraków J. Paweł II", "gates": 6, "lat": 50.077, "lon": 19.784},
    "KTW": {"name": "Katowice-Pyrzowice", "gates": 6, "lat": 50.474, "lon": 19.080},
    "GDN": {"name": "Gdańsk im. Wałęsy", "gates": 5, "lat": 54.377, "lon": 18.466},
    "WRO": {"name": "Wrocław Strachowice", "gates": 5, "lat": 51.102, "lon": 16.898},
    "POZ": {"name": "Poznań-Ławica", "gates": 4, "lat": 52.421, "lon": 16.826},
    "RZE": {"name": "Rzeszów-Jasionka", "gates": 3, "lat": 50.110, "lon": 22.019},
    "LCJ": {"name": "Łódź W. Reymonta", "gates": 3, "lat": 51.722, "lon": 19.398},
    "SZZ": {"name": "Szczecin-Goleniów", "gates": 2, "lat": 53.584, "lon": 14.902},
    "BZG": {"name": "Bydgoszcz Paderewskiego", "gates": 2, "lat": 53.097, "lon": 17.977},
    "LUZ": {"name": "Lublin", "gates": 2, "lat": 51.240, "lon": 22.713},
    "SZY": {"name": "Olsztyn-Mazury", "gates": 2, "lat": 53.481, "lon": 20.937},
    "IEG": {"name": "Zielona Góra-Babimost", "gates": 1, "lat": 52.138, "lon": 15.798},
}

pos = {code: (data["lon"], data["lat"]) for code, data in AIRPORTS.items()}

ROUTES = [
    ("WAW", "KRK", 252, 10), ("WAW", "KTW", 295, 8), ("WAW", "GDN", 340, 10),
    ("WAW", "WRO", 347, 8),  ("WAW", "POZ", 310, 8), ("WAW", "RZE", 295, 4),
    ("WAW", "SZZ", 520, 4),  ("KRK", "GDN", 550, 4), ("KRK", "WRO", 267, 3),
    ("WRO", "GDN", 380, 3),  ("POZ", "KRK", 335, 3), ("KTW", "SZZ", 480, 2),
    ("WAW", "LCJ", 120, 5),  ("WAW", "LUZ", 160, 5), ("WAW", "BZG", 230, 4),
    ("WAW", "SZY", 150, 3),  ("POZ", "IEG", 110, 2), ("GDN", "BZG", 150, 3),
    ("LUZ", "RZE", 140, 2),  ("LCJ", "WRO", 180, 3)
]

FLIGHT_LOGS = []
NOTIFICATIONS = []
AIRPORT_STATUS = {code: {"is_open": True} for code in AIRPORTS.keys()}

G_net = nx.Graph()
G_net.add_nodes_from(pos.keys())
for s, d, dist, f in ROUTES:
    G_net.add_edge(s, d, travel_min=round((dist / CRUISE_SPEED_KMH) * 60), base_freq=f)

def log_notification(time, text):
    NOTIFICATIONS.append({"time": time, "text": text})

def find_backup(current_pos):
    x1, y1 = pos[current_pos]
    closest_node = None
    min_dist_km = float('inf')

    for node, coords in pos.items():
        if AIRPORT_STATUS[node]['is_open'] and node != current_pos:
            dx = (coords[0] - x1) * 70
            dy = (coords[1] - y1) * 111
            dist_km = (dx**2 + dy**2)**0.5
            if dist_km < min_dist_km:
                min_dist_km = dist_km
                closest_node = node

    if closest_node:
        extra_time = round((min_dist_km / CRUISE_SPEED_KMH) * 60)
        return closest_node, max(5, extra_time)
    return current_pos, 0

# =============================================================================
# NAPRAWIONA LOGIKA KOLEJKOWANIA
# =============================================================================
def flight_process(env, f_id, src, dst, airports_res):
    # KLUCZOWA ZMIANA: Rejestrujemy samolot OD RAZU jako uziemiony (start = inf).
    # Dzięki temu będzie widoczny na mapie jako "w kolejce" nawet jeśli nigdy nie wystartuje!
    flight_data = {
        "id": f_id, "src": src, "dst": dst,
        "created": env.now, "start": float('inf'), "end": float('inf'),
        "type": "normal"
    }
    FLIGHT_LOGS.append(flight_data)

    while not AIRPORT_STATUS[src]['is_open']:
        yield env.timeout(3)

    with airports_res[src].request() as req:
        yield req
        yield env.timeout(random.uniform(10, 20))

    # Samolot dostał bramkę i startuje, więc aktualizujemy istniejący wpis
    flight_data["start"] = env.now
    travel_time = G_net[src][dst]['travel_min']
    flight_data["end"] = env.now + travel_time

    yield env.timeout(travel_time)

    if not AIRPORT_STATUS[dst]['is_open']:
        backup_dst, extra_time = find_backup(dst)
        if backup_dst != dst:
            log_notification(env.now, f"⚠️ PRZEKIEROWANIE: Lot {f_id} kierunek {backup_dst}!")
            FLIGHT_LOGS.append({
                "id": f_id, "src": dst, "dst": backup_dst,
                "created": env.now, "start": env.now, "end": env.now + extra_time,
                "type": "rerouted"
            })
            yield env.timeout(extra_time)
            dst = backup_dst

    with airports_res[dst].request() as req:
        yield req
        yield env.timeout(random.uniform(10, 15))

def route_generator(env, src, dst, airports_res):
    freq = G_net[src][dst]['base_freq'] * TRAFFIC_MULTIPLIER
    mean_interval = SIM_DURATION_MIN / freq
    idx = 1
    while True:
        yield env.timeout(random.expovariate(1.0 / mean_interval))
        env.process(flight_process(env, f"{src}{dst[:1]}{idx}", src, dst, airports_res))
        idx += 1

def anomaly_trigger(env, target, start, duration):
    yield env.timeout(start)
    AIRPORT_STATUS[target]['is_open'] = False
    log_notification(env.now, f"❌ BLOKADA: Lotnisko {target} ZAMKNIĘTE! ({AIRPORTS[target]['gates']} bramek odciętych)")
    yield env.timeout(duration)
    AIRPORT_STATUS[target]['is_open'] = True
    log_notification(env.now, f"🟢 REAKTYWACJA: Lotnisko {target} OTWARTE!")

env = simpy.Environment()
airports_res = {code: simpy.Resource(env, capacity=data["gates"]) for code, data in AIRPORTS.items()}

for u, v in G_net.edges():
    env.process(route_generator(env, u, v, airports_res))
    env.process(route_generator(env, v, u, airports_res))

for target_air, start_t, dur_t in ANOMALIES_CONFIG:
    env.process(anomaly_trigger(env, target_air, start_t, dur_t))

print("1. Symulowanie ruchu, generowanie kolejek i zatorów...")
env.run(until=SIM_DURATION_MIN)

# =============================================================================
# WIZUALIZACJA
# =============================================================================
print("2. Renderowanie animacji...")
fig = plt.figure(figsize=(12, 10), facecolor='#111116')
gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
ax_map = fig.add_subplot(gs[0])
ax_txt = fig.add_subplot(gs[1])

persistent_normal_edges = set()
persistent_rerouted_edges = set()
time_steps = np.arange(0, SIM_DURATION_MIN, 3)

def animate(frame):
    current_time = time_steps[frame]
    ax_map.clear(); ax_txt.clear()

    waiting_counts = {node: 0 for node in pos.keys()}
    for leg in FLIGHT_LOGS:
        # Teraz liczenie uwzględnia loty w stanie start = inf
        if leg["created"] <= current_time < leg["start"]:
            waiting_counts[leg["src"]] += 1

        if leg["end"] <= current_time and leg["start"] != float('inf'):
            edge = tuple(sorted([leg["src"], leg["dst"]]))
            if leg["type"] == "normal": persistent_normal_edges.add(edge)
            else: persistent_rerouted_edges.add(edge)

    nx.draw_networkx_edges(G_net, pos, ax=ax_map, edge_color="#222233", style="--", width=0.8)
    nx.draw_networkx_edges(G_net, pos, ax=ax_map, edgelist=list(persistent_normal_edges), edge_color="#00ff66", width=1.5, alpha=0.3)
    nx.draw_networkx_edges(G_net, pos, ax=ax_map, edgelist=list(persistent_rerouted_edges), edge_color="#ff3333", width=3.0, alpha=0.8)

    n_x, n_y, r_x, r_y = [], [], [], []
    for leg in FLIGHT_LOGS:
        if leg["start"] <= current_time < leg["end"] and leg["start"] != float('inf'):
            ratio = (current_time - leg["start"]) / (leg["end"] - leg["start"])
            p_src, p_dst = pos[leg["src"]], pos[leg["dst"]]
            x = p_src[0] + ratio * (p_dst[0] - p_src[0])
            y = p_src[1] + ratio * (p_dst[1] - p_src[1])
            if leg["type"] == "normal": n_x.append(x); n_y.append(y)
            else: r_x.append(x); r_y.append(y)

    if n_x: ax_map.scatter(n_x, n_y, color="#00ff66", marker="^", s=60, zorder=5)
    if r_x: ax_map.scatter(r_x, r_y, color="#ff3333", marker="v", s=100, zorder=6)

    active_closed = set()
    for n_event in NOTIFICATIONS:
        if n_event["time"] <= current_time:
            if "ZAMKNIĘTE" in n_event["text"]: active_closed.add(n_event["text"].split("Lotnisko ")[1].split(" ")[0])
            if "OTWARTE" in n_event["text"]: active_closed.discard(n_event["text"].split("Lotnisko ")[1].split(" ")[0])

    c_open, c_closed = [], []
    for node in pos.keys():
        if node in active_closed: c_closed.append(node)
        else: c_open.append(node)

    node_sizes_open = [AIRPORTS[node]["gates"] * 80 for node in c_open]
    node_sizes_closed = [AIRPORTS[node]["gates"] * 100 for node in c_closed]

    if c_open:
        nx.draw_networkx_nodes(G_net, pos, ax=ax_map, nodelist=c_open, node_color="#1f1f2e", node_size=node_sizes_open, edgecolors="#00ff66", linewidths=1.2)
    if c_closed:
        nx.draw_networkx_nodes(G_net, pos, ax=ax_map, nodelist=c_closed, node_color="#ff3333", node_size=node_sizes_closed, edgecolors="#ffffff", linewidths=1.8)
        for c_node in c_closed:
            ax_map.text(pos[c_node][0], pos[c_node][1] - 0.35, "BLOKADA!", color="#ff3333", weight="bold", ha="center", fontsize=8, bbox=dict(facecolor='black', alpha=0.8, boxstyle='round,pad=0.2'))

    for node, count in waiting_counts.items():
        if count > 0:
            draw_count = min(count, 8)
            xs = [pos[node][0]] * draw_count
            ys = [pos[node][1] + 0.12 + i * 0.05 for i in range(draw_count)]
            ax_map.scatter(xs, ys, color="#ffaa00", marker="s", s=25, zorder=8)
            ax_map.text(pos[node][0], pos[node][1] + 0.12 + draw_count * 0.05 + 0.04,
                        f"{count} na płycie", color="#ffaa00", weight="bold", ha="center", fontsize=8)

    nx.draw_networkx_labels(G_net, pos, ax=ax_map, font_color="#ffffff", font_size=8, font_weight="bold")

    h, m = int(current_time // 60), int(current_time % 60)
    ax_map.set_title(f"MONITOR KRYZYSOWY (ASYMETRYCZNA PRZEPUSTOWOŚĆ) — {h:02d}:{m:02d}", color="#ffffff", fontsize=12, weight="bold")
    ax_map.set_facecolor('#0b0b0f'); ax_map.axis("off")

    ax_txt.set_facecolor('#15151f')
    ax_txt.get_xaxis().set_visible(False); ax_txt.get_yaxis().set_visible(False)
    live_alerts = [n["text"] for n in NOTIFICATIONS if n["time"] <= current_time][-4:]

    y_pos = 0.8
    ax_txt.text(0.02, 0.9, "KOMUNIKATY KONTROLI RUCHU (ATC LIVE):", color="#8888aa", weight="bold", fontsize=10)
    for alert in reversed(live_alerts):
        color = "#ff3333" if "⚠️" in alert or "❌" in alert else "#00ff66"
        ax_txt.text(0.02, y_pos - 0.2, f"» {alert}", color=color, fontsize=9, weight="medium")
        y_pos -= 0.22

ani = animation.FuncAnimation(fig, animate, frames=len(time_steps), interval=70, repeat=False)
plt.tight_layout()

ani.save('zaawansowana_symulacja.gif', writer='pillow', fps=14)
plt.close()
print("🎯 Sukces! Odpal i zobacz jak na Okęciu kumulują się opóźnione loty.")