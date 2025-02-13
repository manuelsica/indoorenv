# Sviluppo di un Agente per la Navigazione Indoor
## Primo Progetto - Intelligenza Artificiale

Questo progetto si concentra sullo sviluppo e l'addestramento di un agente intelligente per la navigazione autonoma in un ambiente 2D con ostacoli e target. L'obiettivo è quello di progettare un sistema che permetta all'agente di prendere decisioni ottimali, massimizzando l'efficienza del percorso e riducendo il rischio di collisioni, tramite l'impiego di algoritmi di apprendimento per rinforzo.

---

## Contenuti del Progetto

- **Documentazione:**  
  Il file `Indoor_navigation.pdf` contiene una descrizione dettagliata dell'implementazione, suddivisa in sezioni che illustrano:
  - La creazione dell'ambiente 2D con `pygame` e `gym`.
  - Le metodologie adottate nei due approcci (parzialmente osservabile vs. completamente osservabile).
  - Le implementazioni degli algoritmi: Q-Learning, SARSA e DQN (con architettura Dueling Double DQN).
  - L'analisi comparativa delle performance e le sfide riscontrate (ad es., problemi di loop e gestione multi-agente).

- **Codice:**  
  Il file `main.py` (insieme ad altri eventuali script) implementa il secondo approccio. In particolare:
  - **Ambiente 2D:** Gestito dalla classe `GridEnvironment` che modella la griglia, l'agente, il target e gli ostacoli.
  - **Algoritmi:**  
    - `QLearningAgent`: Implementa il Q-Learning tabellare.
    - `DQNAgent`: Utilizza una rete neurale Dueling Double DQN per approssimare la funzione Q.
  - **Interfaccia grafica:** Utilizza `pygame` per visualizzare l'ambiente e mostrare in tempo reale le statistiche (episodi, reward, tasso di successo, FPS).

---

## Requisiti

- **Python 3.x**
- **Librerie:**  
  - `pygame`
  - `numpy`
  - `torch` (PyTorch)
  - `matplotlib`
  - `pickle` (standard in Python)

Installa le dipendenze tramite pip:

```bash
pip install pygame numpy torch matplotlib
