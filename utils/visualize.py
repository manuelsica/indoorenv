import matplotlib.pyplot as plt
import numpy as np

def visualize(env, path=None):
    """
    Visualizza l'ambiente e il percorso dell'agente.

    :param env: Ambiente di navigazione.
    :param path: Percorso seguito dall'agente (lista di posizioni).
    """
    grid_size = env.grid_size

    # Crea una matrice per rappresentare la griglia
    grid = np.zeros((grid_size, grid_size))

    # Posiziona gli ostacoli
    for obs in env.obstacles:
        grid[obs[1], obs[0]] = -1  # Ostacolo

    # Posiziona il percorso se fornito
    if path:
        for pos in path:
            if not (np.array_equal(pos, env.agent_pos) or np.array_equal(pos, env.target_pos)):
                grid[pos[1], pos[0]] = 0.5  # Percorso

    # Posiziona il target
    grid[env.target_pos[1], env.target_pos[0]] = 1  # Target

    # Posiziona l'agente
    grid[env.agent_pos[1], env.agent_pos[0]] = 0.75  # Agente

    # Creazione della figura
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='coolwarm', origin='upper')
    plt.colorbar(ticks=[-1, -0.5, 0, 0.5, 0.75, 1], fraction=0.046, pad=0.04)
    plt.clim(-1, 1)
    plt.xticks(np.arange(0, grid_size, 1))
    plt.yticks(np.arange(0, grid_size, 1))
    plt.grid(True, which='both', color='black', linewidth=0.5)
    plt.title('Visualizzazione dell\'Ambiente')
    plt.show()
