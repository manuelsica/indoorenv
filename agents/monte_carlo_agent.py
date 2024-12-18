import numpy as np
from collections import defaultdict


class MonteCarloAgent:
    def __init__(self, action_space, state_space, gamma=0.95,
                 epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.action_space = action_space
        self.state_space = state_space
        self.gamma = gamma  # Fattore di sconto
        self.epsilon = epsilon  # Probabilità di esplorazione
        self.epsilon_decay = epsilon_decay  # Tasso di decadimento di epsilon
        self.min_epsilon = min_epsilon  # Valore minimo di epsilon

        # Q-table inizializzata come defaultdict di array di zeri
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))
        self.returns_sum = defaultdict(float)  # Somma delle ricompense per stato-azione
        self.returns_count = defaultdict(int)  # Conteggio delle visite per stato-azione

    def get_action(self, state):
        """
        Seleziona un'azione utilizzando la politica epsilon-greedy.
        """
        state_key = tuple(state.tolist())
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Esplorazione
        else:
            return np.argmax(self.Q[state_key])  # Sfruttamento

    def record(self, state, action, reward):
        """
        Registra le transizioni per l'episodio corrente.
        """
        if not hasattr(self, 'episode_memory'):
            self.episode_memory = []
        self.episode_memory.append((state, action, reward))

    def update(self):
        """
        Aggiorna la Q-table utilizzando il metodo First-Visit Monte Carlo.
        """
        G = 0
        visited = {}
        # Calcola le ricompense totali dalla fine all'inizio
        for i in reversed(range(len(self.episode_memory))):
            state, action, reward = self.episode_memory[i]
            G = self.gamma * G + reward
            state_key = tuple(state.tolist())
            if state_key not in visited:
                visited[state_key] = action
                self.returns_sum[(state_key, action)] += G
                self.returns_count[(state_key, action)] += 1
                self.Q[state_key][action] = self.returns_sum[(state_key, action)] / self.returns_count[
                    (state_key, action)]

        # Resetta la memoria per il prossimo episodio
        self.episode_memory = []

    def decay_epsilon(self):
        """
        Applica il decadimento di epsilon, mantenendo il valore minimo.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
