import numpy as np
from collections import defaultdict

class SARSAAgent:
    def __init__(self, action_space, state_space, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.action_space = action_space
        self.state_space = state_space
        self.alpha = alpha  # Tasso di apprendimento
        self.gamma = gamma  # Fattore di sconto
        self.epsilon = epsilon  # Probabilità di esplorazione
        self.epsilon_decay = epsilon_decay  # Tasso di decadimento di epsilon
        self.min_epsilon = min_epsilon  # Valore minimo di epsilon

        # Q-table inizializzata come defaultdict di array di zeri
        self.Q = defaultdict(lambda: np.zeros(self.action_space.n))

    def get_action(self, state):
        """
        Seleziona un'azione utilizzando la politica epsilon-greedy.

        :param state: Stato corrente (array numpy).
        :return: Azione selezionata.
        """
        state_key = tuple(state.tolist())  # Converti lo stato in una tupla per usarlo come chiave
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Esplorazione
        else:
            return np.argmax(self.Q[state_key])  # Sfruttamento

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Aggiorna la Q-table utilizzando l'algoritmo SARSA.

        :param state: Stato corrente (array numpy).
        :param action: Azione corrente.
        :param reward: Ricompensa ricevuta.
        :param next_state: Stato successivo (array numpy).
        :param next_action: Prossima azione selezionata.
        :param done: Flag che indica se l'episodio è terminato.
        """
        state_key = tuple(state.tolist())
        next_state_key = tuple(next_state.tolist())
        current_q = self.Q[state_key][action]
        next_q = self.Q[next_state_key][next_action]
        target = reward + self.gamma * next_q * (not done)
        self.Q[state_key][action] += self.alpha * (target - current_q)

    def decay_epsilon_func(self):
        """
        Applica il decadimento di epsilon, mantenendo il valore minimo.
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.min_epsilon)
