import pygame
import sys
import pickle
import numpy as np
from environments.indoor_env import IndoorNavigationEnv
from agents.sarsa_agent import SARSAAgent
from agents.q_learning_agent import QLearningAgent
from collections import defaultdict

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

def interactive_run_agent(agent_type='sarsa', grid_size=10, num_obstacles=10, fixed_obstacles=None, min_distance=2):
    """
    Esegue un agente in modalità interattiva con visualizzazione grafica.

    :param agent_type: Tipo di agente ('sarsa' o 'q_learning').
    :param grid_size: Dimensione della griglia.
    :param num_obstacles: Numero di ostacoli.
    :param fixed_obstacles: Lista di ostacoli fissi.
    :param min_distance: Distanza minima tra ostacoli e tra ostacoli e agenti/target.
    """
    # Inizializza l'ambiente
    env = IndoorNavigationEnv(grid_size=grid_size, num_obstacles=num_obstacles,
                              fixed_obstacles=fixed_obstacles, min_distance=min_distance)

    action_space = env.action_space
    state_space = env.observation_space

    # Inizializza l'agente
    if agent_type == 'sarsa':
        agent = SARSAAgent(action_space=action_space, state_space=state_space)
    elif agent_type == 'q_learning':
        agent = QLearningAgent(action_space=action_space, state_space=state_space)
    else:
        raise ValueError("Tipo di agente non supportato.")

    # Carica la Q-table
    agent.Q = load_q_table(agent_type, action_space.n)
    if agent.Q is None:
        print("Impossibile procedere senza una Q-table valida.")
        return

    # Imposta epsilon a 0 per sfruttamento
    agent.epsilon = 0.0

    # Inizializza Pygame
    pygame.init()
    cell_size = 40
    window_size = grid_size * cell_size
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("Navigazione Indoor con RL")
    clock = pygame.time.Clock()

    # Colori
    COLOR_BG = (255, 255, 255)
    COLOR_AGENT = (0, 0, 255)
    COLOR_TARGET = (255, 0, 0)
    COLOR_OBSTACLE = (0, 0, 0)
    COLOR_GRID = (200, 200, 200)

    # Font
    font = pygame.font.SysFont(None, 24)

    running = True
    agent_running = False
    state = env.reset()
    done = False
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Controlli da tastiera
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    agent_running = not agent_running  # Avvia o ferma l'agente
                    if agent_running:
                        print("Agente avviato.")
                    else:
                        print("Agente fermato.")
                elif event.key == pygame.K_ESCAPE:
                    running = False

        if agent_running and not done:
            state_key = tuple(state.tolist())
            q_values = agent.Q.get(state_key, np.zeros(action_space.n))
            action = np.argmax(q_values)
            next_state, reward, done, info = env.step(action)
            state = next_state
            steps += 1

            # Log delle azioni e dei valori Q
            print(f"Stato corrente: {state}")
            print(f"Valori Q: {q_values}")
            print(f"Azione scelta: {action}")
            print(f"Reward ricevuto: {reward}")

            if done:
                if reward > 0:
                    print("L'agente ha raggiunto il target!")
                else:
                    print("L'agente ha colliso con un ostacolo!")

                # Resetta l'ambiente mantenendo gli ostacoli
                # Genera una nuova posizione target
                while True:
                    target_x = np.random.randint(0, grid_size)
                    target_y = np.random.randint(0, grid_size)
                    target_pos = np.array([target_x, target_y])
                    if not any(np.array_equal(target_pos, obs) for obs in env.obstacles) and not np.array_equal(target_pos, env.agent_pos):
                        break

                state = env.reset(target_pos=target_pos)
                done = False
                steps = 0
                print(f"Nuovo target impostato: {env.target_pos}")

        # Disegna lo sfondo
        screen.fill(COLOR_BG)

        # Disegna la griglia
        for x in range(0, window_size, cell_size):
            pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, window_size))
        for y in range(0, window_size, cell_size):
            pygame.draw.line(screen, COLOR_GRID, (0, y), (window_size, y))

        # Disegna gli ostacoli
        for obs in env.obstacles:
            rect = pygame.Rect(obs[0]*cell_size, obs[1]*cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, COLOR_OBSTACLE, rect)

        # Disegna il target
        rect = pygame.Rect(env.target_pos[0]*cell_size, env.target_pos[1]*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, COLOR_TARGET, rect)

        # Disegna l'agente
        rect = pygame.Rect(env.agent_pos[0]*cell_size, env.agent_pos[1]*cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, COLOR_AGENT, rect)

        # Aggiorna la finestra
        pygame.display.flip()
        clock.tick(2)  # Riduci il valore per rallentare l'agente (ad esempio, 2 FPS)

    pygame.quit()
    sys.exit()
