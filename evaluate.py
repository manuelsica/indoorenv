import gym
from environments.indoor_env import IndoorNavigationEnv
from agents.sarsa_agent import SARSAAgent
from agents.q_learning_agent import QLearningAgent
import numpy as np
import pickle  # Per caricare le Q-table
from utils.visualize import visualize
from collections import defaultdict
import os

def load_q_table(agent_type, action_space_n):
    """
    Carica la Q-table da disco.

    :param agent_type: Tipo di agente ('sarsa' o 'q_learning').
    :param action_space_n: Numero di azioni disponibili.
    :return: Q-table caricata come defaultdict.
    """
    try:
        with open(f"q_tables/{agent_type}_q_table.pkl", 'rb') as f:
            Q = pickle.load(f)
        print(f"Q-table per l'agente {agent_type} caricata con successo.")
        # Reconverte in defaultdict con array di zeri
        Q_default = defaultdict(lambda: np.zeros(action_space_n), Q)
        return Q_default
    except FileNotFoundError:
        print(f"Errore: Q-table per l'agente {agent_type} non trovata.")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento della Q-table: {e}")
        return None

def evaluate_agent(agent_type='sarsa', episodes=100, grid_size=15, num_obstacles=10, fixed_obstacles=None, min_distance=2):
    """
    Valuta un agente SARSA o Q-Learning.

    :param agent_type: Tipo di agente ('sarsa' o 'q_learning').
    :param episodes: Numero di episodi di valutazione.
    :param grid_size: Dimensione della griglia dell'ambiente.
    :param num_obstacles: Numero di ostacoli nell'ambiente.
    :param fixed_obstacles: Lista di posizioni fisse per gli ostacoli (opzionale).
    :param min_distance: Distanza minima tra ostacoli e tra ostacoli e agenti/target.
    """
    # Crea l'ambiente con parametri aggiornati
    env = IndoorNavigationEnv(grid_size=grid_size, num_obstacles=num_obstacles,
                              fixed_obstacles=fixed_obstacles, min_distance=min_distance)
    state_space = env.observation_space
    action_space = env.action_space

    # Inizializza l'agente
    if agent_type == 'sarsa':
        agent = SARSAAgent(action_space=action_space, state_space=state_space)
    elif agent_type == 'q_learning':
        agent = QLearningAgent(action_space=action_space, state_space=state_space)
    else:
        raise ValueError("Tipo di agente non supportato.")

    # Carica la Q-table addestrata
    agent.Q = load_q_table(agent_type, action_space.n)
    if agent.Q is None:
        print("Impossibile procedere con la valutazione senza una Q-table valida.")
        return

    total_rewards = 0  # Ricompensa totale su tutti gli episodi
    total_steps = 0    # Passi totali su tutti gli episodi
    success = 0        # Numero di successi (agente ha raggiunto il target)

    for episode in range(1, episodes + 1):
        # Seleziona una posizione target casuale
        while True:
            target_x = np.random.randint(0, grid_size)
            target_y = np.random.randint(0, grid_size)
            target_pos = np.array([target_x, target_y])
            if not any(np.array_equal(target_pos, obs) for obs in env.obstacles) and not np.array_equal(target_pos, env.agent_pos):
                break

        state = env.reset(target_pos=target_pos, keep_obstacles=True)
        done = False
        steps = 0
        path = [env.agent_pos.copy()]  # Traccia del percorso dell'agente

        while not done and steps < 100:
            state_key = tuple(state.tolist())
            action = np.argmax(agent.Q.get(state_key, np.zeros(action_space.n)))
            next_state, reward, done, _ = env.step(action)
            steps += 1
            path.append(env.agent_pos.copy())
            state = next_state

        total_rewards += reward
        total_steps += steps
        if reward > 0:
            success +=1

        # Visualizzazione dell'ultimo episodio
        if episode == episodes:
            print(f"\nVisualizzazione dell'ultimo episodio per l'agente {agent_type}:")
            visualize(env, path=path)

        # Informazioni per ogni episodio
        print(f"Episodio {episode}/{episodes} completato. Ricompensa: {reward}, Passi: {steps}, Successo: {'Sì' if reward > 0 else 'No'}")

    # Calcola e stampa i risultati della valutazione
    print(f"\n--- Risultati della Valutazione per l'agente {agent_type} ---")
    print(f"Epoche: {episodes}")
    print(f"Ricompensa Media: {total_rewards / episodes:.2f}")
    print(f"Passi Medi: {total_steps / episodes:.2f}")
    print(f"Successi: {success} / {episodes}")
