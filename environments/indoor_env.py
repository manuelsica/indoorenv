import numpy as np
import gym
from gym import spaces
from collections import deque

class IndoorNavigationEnv(gym.Env):
    """
    Ambiente di navigazione indoor 2D personalizzato.
    L'agente deve navigare da una posizione di partenza a una posizione target,
    evitando ostacoli.
    """

    def __init__(self, grid_size=15, num_obstacles=10, fixed_obstacles=None, min_distance=2):
        super(IndoorNavigationEnv, self).__init__()
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.min_distance = min_distance  # Distanza minima tra ostacoli e dall'agente/target

        # Definizione dello spazio degli stati
        # Stato: [dx_target, dy_target, obs_N, obs_S, obs_E, obs_W]
        self.observation_space = spaces.Box(low=-grid_size+1, high=grid_size-1,
                                            shape=(6,), dtype=np.int32)

        # Definizione dello spazio delle azioni
        # 0: Su, 1: Giù, 2: Sinistra, 3: Destra
        self.action_space = spaces.Discrete(4)

        # Inizializzazione delle posizioni
        self.agent_pos = np.array([0, 0])  # Posizione iniziale dell'agente
        self.target_pos = np.array([grid_size-1, grid_size-1])  # Posizione target predefinita
        self.obstacles = []  # Lista di posizioni degli ostacoli

        # Ostacoli fissi se forniti
        self.fixed_obstacles = fixed_obstacles

        # Flag per determinare se gli ostacoli sono fissi
        self.obstacles_fixed = False

        # Insieme delle posizioni visitate
        self.visited_positions = set()

        # Resetta l'ambiente
        self.reset()

    def reset(self, target_pos=None, keep_obstacles=True):
        """
        Resetta l'ambiente per un nuovo episodio.

        :param target_pos: Lista o array di due elementi [x, y] per la posizione del target.
                           Se None, usa la posizione target predefinita.
        :param keep_obstacles: Se True, non rigenera gli ostacoli.
        :return: Stato iniziale.
        """
        self.agent_pos = np.array([0, 0])  # Reset posizione agente
        if target_pos is not None:
            self.target_pos = np.array(target_pos)  # Imposta posizione target
        else:
            self.target_pos = np.array([self.grid_size-1, self.grid_size-1])  # Posizione target predefinita

        self.visited_positions = set()
        self.visited_positions.add(tuple(self.agent_pos))

        if not keep_obstacles or not self.obstacles_fixed:
            # Genera ostacoli solo se non sono stati fissati o se keep_obstacles è False
            if self.fixed_obstacles:
                # Se sono stati forniti ostacoli fissi, usali
                self.obstacles = [np.array(pos) for pos in self.fixed_obstacles]
            else:
                # Altrimenti, genera ostacoli casuali
                self.obstacles = []
                attempts = 0
                max_attempts = self.num_obstacles * 50  # Aumenta i tentativi per posizionare ostacoli distribuiti
                while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
                    pos = np.random.randint(0, self.grid_size, size=2)
                    pos = np.array(pos)
                    # Calcola la distanza dalla posizione dell'agente e del target
                    distance_to_agent = np.linalg.norm(pos - self.agent_pos)
                    distance_to_target = np.linalg.norm(pos - self.target_pos)

                    # Calcola la distanza dagli altri ostacoli
                    distances_to_obstacles = [np.linalg.norm(pos - obs) for obs in self.obstacles]
                    min_distance_to_obstacles = min(distances_to_obstacles) if distances_to_obstacles else float('inf')

                    # Verifica se la distanza minima è rispettata
                    if (distance_to_agent >= self.min_distance and
                        distance_to_target >= self.min_distance and
                        min_distance_to_obstacles >= self.min_distance and
                        not any(np.array_equal(pos, obs) for obs in self.obstacles)):
                        self.obstacles.append(pos)
                    attempts +=1

                if len(self.obstacles) < self.num_obstacles:
                    print(f"Avviso: Non è stato possibile posizionare tutti gli ostacoli richiesti. Ostacoli posizionati: {len(self.obstacles)}")

            # Verifica la connettività
            if not self.is_reachable():
                print("Ostacoli bloccano il target. Rigenero gli ostacoli.")
                return self.reset(target_pos=target_pos, keep_obstacles=False)  # Ricomincia il reset

            # Fissa gli ostacoli
            self.obstacles_fixed = True

        return self._get_obs()

    def set_target(self, target_pos):
        """
        Imposta una nuova posizione target senza rigenerare gli ostacoli.
        Resetta l'agente alla posizione iniziale.
        :param target_pos: Lista o array di due elementi [x, y] per la posizione del target.
        :return: Stato iniziale.
        """
        self.agent_pos = np.array([0, 0])  # Reset posizione agente
        self.target_pos = np.array(target_pos)
        self.visited_positions = set()
        self.visited_positions.add(tuple(self.agent_pos))
        return self._get_obs()

    def is_reachable(self):
        """
        Verifica se il target è raggiungibile dall'agente utilizzando BFS.
        :return: True se raggiungibile, False altrimenti.
        """
        queue = deque()
        queue.append(tuple(self.agent_pos))
        visited = set()
        visited.add(tuple(self.agent_pos))
        target = tuple(self.target_pos)

        while queue:
            current = queue.popleft()
            if current == target:
                return True

            # Definisci i movimenti possibili (su, giù, sinistra, destra)
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for move in moves:
                neighbor = (current[0] + move[0], current[1] + move[1])
                if (0 <= neighbor[0] < self.grid_size and
                    0 <= neighbor[1] < self.grid_size and
                    neighbor not in visited and
                    not any(np.array_equal(neighbor, obs) for obs in self.obstacles)):
                    queue.append(neighbor)
                    visited.add(neighbor)

        return False

    def step(self, action):
        """
        Esegue un passo nell'ambiente.

        :param action: Azione da eseguire (0: Su, 1: Giù, 2: Sinistra, 3: Destra).
        :return: Tuple (next_state, reward, done, info).
        """
        previous_pos = self.agent_pos.copy()  # Memorizza la posizione precedente

        # Mappa l'azione a un movimento
        movement = np.array([0, 0])
        if action == 0:  # Su
            movement = np.array([0, -1])
        elif action == 1:  # Giù
            movement = np.array([0, 1])
        elif action == 2:  # Sinistra
            movement = np.array([-1, 0])
        elif action == 3:  # Destra
            movement = np.array([1, 0])

        # Calcola la nuova posizione
        new_pos = self.agent_pos + movement

        # Verifica i confini della griglia
        if (new_pos[0] < 0 or new_pos[0] >= self.grid_size or
            new_pos[1] < 0 or new_pos[1] >= self.grid_size):
            # Movimento non valido, rimani nella posizione attuale
            reward = -5  # Penalità per tentativo di uscire dalla griglia
            done = False
            return self._get_obs(), reward, done, {}

        # Verifica collisioni con ostacoli
        if any(np.array_equal(new_pos, obs) for obs in self.obstacles):
            reward = -100  # Penalità per collisione
            done = True  # Episodio terminato
            return self._get_obs(), reward, done, {}
        else:
            # Aggiorna la posizione dell'agente
            self.agent_pos = new_pos

            # Controlla se la posizione è stata già visitata
            position_tuple = tuple(self.agent_pos)
            if position_tuple in self.visited_positions:
                revisit_penalty = -10  # Penalità per aver rivisitato una posizione
            else:
                revisit_penalty = 0
                self.visited_positions.add(position_tuple)

            # Reward shaping migliorato
            if np.array_equal(self.agent_pos, self.target_pos):
                reward = 100  # Ricompensa per aver raggiunto il target
                done = True  # Episodio terminato
            else:
                # Calcola la variazione della distanza
                distance_before = np.linalg.norm(previous_pos - self.target_pos)
                distance_after = np.linalg.norm(self.agent_pos - self.target_pos)
                delta_distance = distance_before - distance_after

                if delta_distance > 0:
                    reward = 1  # Ricompensa per avvicinamento
                elif delta_distance == 0:
                    reward = -1  # Penalità per non muoversi verso il target
                else:
                    reward = -3  # Penalità per allontanamento

                # Aggiungi la penalità per la rivisitazione
                reward += revisit_penalty

                done = False  # Episodio continua

            return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """
        Ritorna l'osservazione corrente.
        Stato: [dx_target, dy_target, obs_N, obs_S, obs_E, obs_W]
        """
        # Calcola le distanze relative dall'agente al target
        distance_to_target = self.target_pos - self.agent_pos  # [dx, dy]

        # Definisci una distanza di rilevamento per gli ostacoli
        detection_distance = 2

        # Inizializza le variabili per le direzioni
        obs_N = 0
        obs_S = 0
        obs_E = 0
        obs_W = 0

        for obs in self.obstacles:
            dx = obs[0] - self.agent_pos[0]
            dy = obs[1] - self.agent_pos[1]

            if dx == 0 and 0 < dy <= detection_distance:
                obs_S = 1
            elif dx == 0 and -detection_distance <= dy < 0:
                obs_N = 1
            elif dy == 0 and 0 < dx <= detection_distance:
                obs_E = 1
            elif dy == 0 and -detection_distance <= dx < 0:
                obs_W = 1

        # Concatenazione delle distanze e delle osservazioni degli ostacoli
        obs = np.concatenate((
            distance_to_target,  # [dx_target, dy_target]
            np.array([obs_N, obs_S, obs_E, obs_W])  # [obs_N, obs_S, obs_E, obs_W]
        ))
        return obs

    def render(self, mode='human'):
        """
        Renderizza l'ambiente sul terminale.

        :param mode: Modalità di rendering (attualmente solo 'human').
        """
        # Crea una griglia vuota
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # Posiziona gli ostacoli
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = 'X'

        # Posiziona il target
        grid[self.target_pos[1]][self.target_pos[0]] = 'T'

        # Posiziona l'agente
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'

        # Stampa la griglia
        print("\n".join([" ".join(row) for row in grid]))
        print()  # Linea vuota per separazione
