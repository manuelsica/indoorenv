import gym
from environments.indoor_env import IndoorNavigationEnv
from agents.sarsa_agent import SARSAAgent
from agents.q_learning_agent import QLearningAgent
import numpy as np
import matplotlib.pyplot as plt
import pickle  # Per salvare le Q-table
import random
import os
import logging

def save_q_table(agent_type, Q):
    """
    Salva la Q-table su disco come file pickle.

    :param agent_type: Tipo di agente ('sarsa' o 'q_learning').
    :param Q: Q-table da salvare.
    """
    os.makedirs('q_tables', exist_ok=True)  # Crea la cartella se non esiste
    with open(f"q_tables/{agent_type}_q_table.pkl", "wb") as f:
        pickle.dump(dict(Q), f)  # Converti defaultdict in dict prima di salvare

def train_agent(agent_type='sarsa', episodes=10000, grid_size=15, num_obstacles=10, fixed_obstacles=None, min_distance=2):
    """
    Addestra un agente SARSA o Q-Learning.

    :param agent_type: Tipo di agente ('sarsa' o 'q_learning').
    :param episodes: Numero di episodi di addestramento.
    :param grid_size: Dimensione della griglia dell'ambiente.
    :param num_obstacles: Numero di ostacoli nell'ambiente.
    :param fixed_obstacles: Lista di posizioni fisse per gli ostacoli (opzionale).
    :param min_distance: Distanza minima tra ostacoli e tra ostacoli e agenti/target.
    """
    # Crea l'ambiente di navigazione indoor
    env = IndoorNavigationEnv(grid_size=grid_size, num_obstacles=num_obstacles,
                              fixed_obstacles=fixed_obstacles, min_distance=min_distance)
    state_space = env.observation_space
    action_space = env.action_space

    # Inizializza l'agente
    if agent_type == 'sarsa':
        agent = SARSAAgent(
            action_space=action_space,
            state_space=state_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.9995,
            min_epsilon=0.01
        )
    elif agent_type == 'q_learning':
        agent = QLearningAgent(
            action_space=action_space,
            state_space=state_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.9995,
            min_epsilon=0.01
        )
    else:
        raise ValueError("Tipo di agente non supportato.")

    # Inizializza liste per tracciare ricompense e passi per episodio
    rewards_per_episode = []
    steps_per_episode = []

    # Configura il logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Loop di addestramento per ogni episodio
    for episode in range(1, episodes +1):
        # Seleziona una posizione target casuale che non sia un ostacolo o la posizione dell'agente
        while True:
            target_x = random.randint(0, grid_size-1)
            target_y = random.randint(0, grid_size-1)
            target_pos = np.array([target_x, target_y])
            if not any(np.array_equal(target_pos, obs) for obs in env.obstacles) and not np.array_equal(target_pos, env.agent_pos):
                break

        state = env.reset(target_pos=target_pos, keep_obstacles=True)  # Reset dell'ambiente con target casuale
        action = agent.get_action(state)  # Seleziona azione iniziale

        total_reward = 0  # Ricompensa totale per l'episodio
        steps = 0  # Numero di passi per l'episodio

        # Ciclo per ogni passo nell'episodio
        while True:
            next_state, reward, done, _ = env.step(action)  # Esegui l'azione nell'ambiente
            total_reward += reward
            steps +=1

            if agent_type == 'sarsa':
                next_action = agent.get_action(next_state)  # Seleziona la prossima azione
                agent.update(state, action, reward, next_state, next_action, done)  # Aggiorna la Q-table
                action = next_action  # Passa alla prossima azione
            elif agent_type == 'q_learning':
                agent.update(state, action, reward, next_state, done)  # Aggiorna la Q-table
                if not done:
                    action = agent.get_action(next_state)  # Se non è finito, seleziona nuova azione

            state = next_state  # Aggiorna lo stato

            if done:
                break  # Episodio terminato

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Applica il decadimento di epsilon
        agent.decay_epsilon_func()

        # Salva la Q-table ogni 5000 episodi
        if episode % 5000 == 0:
            save_q_table(agent_type, agent.Q)
            print(f"Q-table salvata per l'agente {agent_type} all'episodio {episode}.")
            logging.info(f"Q-table salvata per l'agente {agent_type} all'episodio {episode}.")

        # Stampa informazioni ogni 500 episodi
        if episode % 500 == 0:
            avg_reward = np.mean(rewards_per_episode[-500:])
            avg_steps = np.mean(steps_per_episode[-500:])
            print(f"Epoca {episode}/{episodes}, Ricompensa Media (500): {avg_reward:.3f}, Passi Medi (500): {avg_steps:.3f}")
            logging.info(f"Epoca {episode}/{episodes}, Ricompensa Media (500): {avg_reward:.3f}, Passi Medi (500): {avg_steps:.3f}")

    # Salva la Q-table addestrata
    save_q_table(agent_type, agent.Q)

    # Plot delle ricompense per episodio
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(rewards_per_episode)
    plt.title(f'Ricompensa per Episodio ({agent_type})')
    plt.xlabel('Episodi')
    plt.ylabel('Ricompensa Totale')

    # Plot dei passi per episodio
    plt.subplot(1,2,2)
    plt.plot(steps_per_episode)
    plt.title(f'Passi per Episodio ({agent_type})')
    plt.xlabel('Episodi')
    plt.ylabel('Numero di Passi')

    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)  # Crea la cartella se non esiste
    plt.savefig(f'plots/{agent_type}_training_plot.png')  # Salva il plot come immagine
    plt.show()  # Mostra il plot

    print(f"Addestramento completato per l'agente {agent_type}.")
    logging.info(f"Addestramento completato per l'agente {agent_type}.")
