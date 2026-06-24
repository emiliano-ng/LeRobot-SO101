import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Carpeta donde está este script
BASE_DIR = Path(__file__).parent
PARQUET_FILE = BASE_DIR / "file-000.parquet"

# Nombres reales de los estados del robot
STATE_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Leer archivo parquet
df = pd.read_parquet(PARQUET_FILE)

print("Columnas del dataset:")
print(df.columns)

print("\nEpisodios encontrados:")
print(df["episode_index"].unique())

print("\nNúmero de episodios:")
print(df["episode_index"].nunique())

# Convertir observation.state de lista a columnas separadas
states = pd.DataFrame(df["observation.state"].tolist(), columns=STATE_NAMES)

# Agregar columnas útiles
states["timestamp"] = df["timestamp"]
states["frame_index"] = df["frame_index"]
states["episode_index"] = df["episode_index"]

print("\nPrimeras filas de estados:")
print(states.head())

print("\nDescripción estadística:")
print(states[STATE_NAMES].describe())

# Guardar CSV para revisar en Excel o VS Code
states.to_csv(BASE_DIR / "states_named.csv", index=False)
print("\nArchivo creado: states_named.csv")

# ==============================
# Gráfica general: todos los estados
# ==============================

plt.figure(figsize=(12, 7))

for name in STATE_NAMES:
    plt.plot(states["timestamp"], states[name], label=name)

plt.title("Robot joint states vs time")
plt.xlabel("Time (s)")
plt.ylabel("Joint position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(BASE_DIR / "states_vs_time.png", dpi=300)
plt.show()

print("Gráfica creada: states_vs_time.png")

# ==============================
# Gráfica por episodio
# ==============================

episodes = states["episode_index"].unique()

for episode in episodes:
    episode_data = states[states["episode_index"] == episode]

    plt.figure(figsize=(12, 7))

    for name in STATE_NAMES:
        plt.plot(
            episode_data["timestamp"],
            episode_data[name],
            label=name
        )

    plt.title(f"Robot joint states vs time - Episode {episode}")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / f"states_episode_{episode}.png", dpi=300)
    plt.show()

    print(f"Gráfica creada: states_episode_{episode}.png")