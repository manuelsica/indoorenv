import pygame
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt

# =============================================================================
# PARAMETRI GLOBALI
# =============================================================================

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
GRID_SIZE = 10
CELL_SIZE = 60
FONT_SIZE = 20

BACKGROUND_COLOR = (230, 240, 255)
GRID_COLOR = (190, 190, 220)
AGENT_COLOR = (50, 205, 50)
TARGET_COLOR = (255, 69, 0)
OBSTACLE_COLOR = (60, 60, 60)
TEXT_COLOR = (30, 30, 30)

FPS = 5

REWARD_REACH_TARGET = 10.0
REWARD_COLLISION = -10.0
REWARD_STEP = -1.0

Q_TABLE_FILE = "q_table.pkl"
DQN_FILE = "dqn_model.pt"

# Limite di step per episodio (evita loop infiniti)
MAX_STEPS_PER_EPISODE = 200

# Penalità se la posizione finale dell'agente = posizione iniziale (no move)
NO_MOVE_PENALTY = -0.2

# Epsilon in inferenza (anche se carichiamo un modello, lasciamo un pizzico di esplorazione)
EPSILON_INFERENZA_MIN = 0.01

# =============================================================================
# CLASSE ENV 2D
# =============================================================================

class GridEnvironment:
    """
    Ambiente 2D:
      - muove l'agente su/giù/sinistra/destra
      - se si raggiunge target => spawn ostacolo
      - se collisione => reset ostacoli
      - no numero episodi definito
      - bottone "Stop training" e +/- FPS
    """
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Ambiente 2D RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        # Pulsanti su schermo
        self.stop_button_rect = pygame.Rect(WINDOW_WIDTH - 120, 10, 110, 30)
        self.fps_plus_rect = pygame.Rect(WINDOW_WIDTH - 50, 50, 40, 30)
        self.fps_minus_rect = pygame.Rect(WINDOW_WIDTH - 100, 50, 40, 30)

        self.stop_training = False
        self.agent_pos = [0, 0]
        self.target_pos = [0, 0]
        self.obstacles = []
        self.targets_reached = 0

        self.reset_environment_state()

    def reset_environment_state(self):
        self.agent_pos = [random.randint(0, GRID_SIZE - 1),
                          random.randint(0, GRID_SIZE - 1)]
        self.target_pos = [random.randint(0, GRID_SIZE - 1),
                           random.randint(0, GRID_SIZE - 1)]
        while self.target_pos == self.agent_pos:
            self.target_pos = [random.randint(0, GRID_SIZE - 1),
                               random.randint(0, GRID_SIZE - 1)]
        self.obstacles = []
        self.targets_reached = 0

    def get_state(self):
        obstacle_hash = 0
        for (ox, oy) in self.obstacles:
            obstacle_hash += (ox + oy * GRID_SIZE + 1)
        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.target_pos[0],
            self.target_pos[1],
            obstacle_hash
        )

    def step(self, action, old_pos=None):
        """
        Ritorna next_state, reward, done.
        old_pos: la posizione dell'agente PRIMA di eseguire l'azione
                 per controllare se si è mosso davvero.
        """
        if action == 0 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < GRID_SIZE - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < GRID_SIZE - 1:
            self.agent_pos[0] += 1

        reward = REWARD_STEP
        done = False

        # Controllo collisione
        if tuple(self.agent_pos) in [tuple(o) for o in self.obstacles]:
            reward += REWARD_COLLISION
            self.targets_reached = 0
            self.obstacles = []
            done = True

        # Controllo se raggiunge target
        if self.agent_pos == self.target_pos:
            reward += REWARD_REACH_TARGET
            self.targets_reached += 1
            self.spawn_obstacle()
            self.place_new_target()
            done = True

        # Se la posizione finale = old_pos => penalità
        if old_pos is not None and old_pos == self.agent_pos:
            reward += NO_MOVE_PENALTY

        return self.get_state(), reward, done

    def spawn_obstacle(self):
        while True:
            ox = random.randint(0, GRID_SIZE - 1)
            oy = random.randint(0, GRID_SIZE - 1)
            if [ox, oy] != self.agent_pos and [ox, oy] != self.target_pos:
                if [ox, oy] not in self.obstacles:
                    self.obstacles.append([ox, oy])
                    break

    def place_new_target(self):
        while True:
            tx = random.randint(0, GRID_SIZE - 1)
            ty = random.randint(0, GRID_SIZE - 1)
            if [tx, ty] != self.agent_pos and [tx, ty] not in self.obstacles:
                self.target_pos = [tx, ty]
                break

    def render(self, ep_count, algo_name,
               avg_r_q, sr_q,
               avg_r_dqn, sr_dqn):
        self.screen.fill(BACKGROUND_COLOR)

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if (x+y)%2==0:
                    cell_col = (220, 230, 240)
                else:
                    cell_col = (235, 240, 255)
                pygame.draw.rect(self.screen, cell_col, rect)
                pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        # Target
        tgt_center = (self.target_pos[0]*CELL_SIZE + CELL_SIZE//2,
                      self.target_pos[1]*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(self.screen, TARGET_COLOR, tgt_center, CELL_SIZE//3)

        # Ostacoli
        for ox, oy in self.obstacles:
            obs_r = pygame.Rect(ox*CELL_SIZE+5, oy*CELL_SIZE+5,
                                CELL_SIZE-10, CELL_SIZE-10)
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, obs_r)

        # Agente
        ag_center = (self.agent_pos[0]*CELL_SIZE + CELL_SIZE//2,
                     self.agent_pos[1]*CELL_SIZE + CELL_SIZE//2)
        pygame.draw.circle(self.screen, AGENT_COLOR, ag_center, CELL_SIZE//3)

        # Bottone Stop
        pygame.draw.rect(self.screen, (200,0,0), self.stop_button_rect)
        stop_txt = self.font.render("Stop", True, (255,255,255))
        self.screen.blit(stop_txt, (self.stop_button_rect.x+20, self.stop_button_rect.y+5))

        # Pulsanti FPS
        pygame.draw.rect(self.screen, (0,150,200), self.fps_plus_rect)
        plus_txt = self.font.render("+", True, (255,255,255))
        self.screen.blit(plus_txt, (self.fps_plus_rect.x+12, self.fps_plus_rect.y+3))

        pygame.draw.rect(self.screen, (0,150,200), self.fps_minus_rect)
        minus_txt = self.font.render("-", True, (255,255,255))
        self.screen.blit(minus_txt, (self.fps_minus_rect.x+12, self.fps_minus_rect.y+3))

        # Info
        info_lines = [
            f"Algoritmo: {algo_name}",
            f"Episodio: {ep_count}",
            f"FPS: {FPS}",
            f"Targets consecutivi: {self.targets_reached}",
            f"Q-Learning Avg Reward: {round(avg_r_q,2)} | SR: {round(sr_q*100,2)}%",
            f"DQN Avg Reward: {round(avg_r_dqn,2)} | SR: {round(sr_dqn*100,2)}%"
        ]
        for i, line in enumerate(info_lines):
            tsurf = self.font.render(line, True, TEXT_COLOR)
            self.screen.blit(tsurf, (10, 10+i*25))

        pygame.display.flip()

    def handle_events(self):
        global FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop_training = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mpos = event.pos
                if self.stop_button_rect.collidepoint(mpos):
                    self.stop_training = True
                elif self.fps_plus_rect.collidepoint(mpos):
                    FPS += 1
                elif self.fps_minus_rect.collidepoint(mpos) and FPS>1:
                    FPS -= 1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    try:
                        new_fps_str = input("Inserisci nuovo FPS: ")
                        new_fps = int(new_fps_str)
                        if new_fps>0:
                            FPS = new_fps
                            print(f"FPS aggiornati a {FPS}")
                    except ValueError:
                        print("Valore non valido per FPS.")

    def close(self):
        pygame.quit()

# =============================================================================
# Q-LEARNING
# =============================================================================

class QLearningAgent:
    """
    Agente Q-Learning tabellare, con epsilon decay.
    In inferenza => epsilon=EPSILON_INFERENZA_MIN >= 0.01
    """
    def __init__(self, inferenza=False):
        self.gamma = 0.99
        self.alpha = 0.1
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.inferenza = inferenza
        self.q_table = {}

        if self.inferenza:
            # Consentiamo un piccolo epsilon anche in inferenza
            self.epsilon = EPSILON_INFERENZA_MIN

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        return self.q_table[state]

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,3)
        else:
            return int(np.argmax(self.get_q_values(state)))

    def update(self, state, action, reward, next_state, done):
        if self.inferenza:
            return
        current_q = self.get_q_values(state)[action]
        max_next_q = 0 if done else np.max(self.get_q_values(next_state))
        new_q = current_q + self.alpha*(reward+self.gamma*max_next_q-current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        if not self.inferenza:
            self.epsilon = max(self.epsilon_min,self.epsilon*self.epsilon_decay)

    def save_model(self):
        with open(Q_TABLE_FILE,'wb') as f:
            pickle.dump(self.q_table,f)

    def load_model(self):
        if os.path.exists(Q_TABLE_FILE):
            with open(Q_TABLE_FILE,'rb') as f:
                self.q_table=pickle.load(f)

# =============================================================================
# DUELING DOUBLE DQN
# =============================================================================

class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5,128)
        self.fc2 = nn.Linear(128,128)
        self.value_stream = nn.Linear(128,1)
        self.adv_stream = nn.Linear(128,4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        val = self.value_stream(x)
        adv = self.adv_stream(x)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q

class DQNAgent:
    """
    DQN con rete Dueling e Double DQN.
    In inferenza => epsilon=EPSILON_INFERENZA_MIN
    """
    def __init__(self, inferenza=False):
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 0.001
        self.batch_size = 64
        self.update_target_steps = 1000

        self.inferenza=inferenza
        if self.inferenza:
            self.epsilon=EPSILON_INFERENZA_MIN

        self.policy_net = DQNetwork()
        self.target_net = DQNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = []
        self.step_count=0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0,3)
        else:
            st_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_vals = self.policy_net(st_t)
            return int(torch.argmax(q_vals,dim=1).item())

    def store_experience(self, state, action, reward, next_state, done):
        if self.inferenza:
            return
        self.memory.append((state,action,reward,next_state,done))
        if len(self.memory)>100000:
            self.memory.pop(0)

    def sample_memory(self):
        import random
        batch = random.sample(self.memory,self.batch_size)
        states, actions, rewards, next_states, dones=zip(*batch)
        return states, actions, rewards, next_states, dones

    def update(self):
        if self.inferenza:
            return
        if len(self.memory)<self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory()

        st_t = torch.FloatTensor(states)
        act_t = torch.LongTensor(actions).unsqueeze(1)
        rew_t = torch.FloatTensor(rewards).unsqueeze(1)
        nst_t = torch.FloatTensor(next_states)
        done_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).unsqueeze(1)

        q_vals = self.policy_net(st_t).gather(1,act_t)

        with torch.no_grad():
            next_actions = self.policy_net(nst_t).argmax(dim=1, keepdim=True)
            q_next_target = self.target_net(nst_t).gather(1,next_actions)

        target_q = rew_t + (1-done_t)*self.gamma*q_next_target
        loss = F.mse_loss(q_vals,target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count%self.update_target_steps==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        if not self.inferenza:
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def save_model(self):
        torch.save(self.policy_net.state_dict(),DQN_FILE)

    def load_model(self):
        if os.path.exists(DQN_FILE):
            self.policy_net.load_state_dict(torch.load(DQN_FILE))
            self.target_net.load_state_dict(self.policy_net.state_dict())

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Scegli l'opzione:")
    print("1) Caricare i modelli addestrati (solo inferenza, con un piccolo epsilon)")
    print("2) Allenare da zero (training completo)")
    choice_mode = input("Inserisci 1 o 2: ")

    print("Scegli algoritmo:")
    print("a) Q-Learning")
    print("b) Dueling Double DQN")
    choice_algo = input("Inserisci a o b: ")

    inference=(choice_mode=="1")

    env = GridEnvironment()

    if choice_algo.lower()=="a":
        agent_name="Q-Learning"
        agent_in_use=QLearningAgent(inferenza=inference)
    else:
        agent_name="DQN"
        agent_in_use=DQNAgent(inferenza=inference)

    if inference:
        agent_in_use.load_model()
        print(f"Caricato modello {agent_name}, solo inferenza con epsilon minimo.")
    else:
        print(f"Allenamento da zero con {agent_name}...")

    # Statistiche
    window_size=50
    rewards_q_window=[]
    successes_q_window=[]
    rewards_dqn_window=[]
    successes_dqn_window=[]

    episodes_q_history=[]
    avg_rewards_q_history=[]
    success_rate_q_history=[]

    episodes_dqn_history=[]
    avg_rewards_dqn_history=[]
    success_rate_dqn_history=[]

    episode_q=0
    episode_dqn=0

    while not env.stop_training:
        env.handle_events()
        env.clock.tick(FPS)

        if agent_name=="Q-Learning":
            total_reward=0
            steps=0
            state=env.get_state()
            done=False
            while not done and not env.stop_training:
                old_pos=list(env.agent_pos)
                action=agent_in_use.select_action(state)
                next_state,reward,done=env.step(action,old_pos=old_pos)
                agent_in_use.update(state,action,reward,next_state,done)
                total_reward+=reward
                state=next_state
                steps+=1

                if steps>MAX_STEPS_PER_EPISODE:
                    done=True

                env.handle_events()
                env.render(
                    episode_q,
                    agent_name,
                    np.mean(rewards_q_window) if rewards_q_window else 0,
                    np.mean(successes_q_window) if successes_q_window else 0,
                    np.mean(rewards_dqn_window) if rewards_dqn_window else 0,
                    np.mean(successes_dqn_window) if successes_dqn_window else 0
                )
                env.clock.tick(FPS)

            episode_q+=1
            rewards_q_window.append(total_reward)
            successes_q_window.append(1 if total_reward>0 else 0)
            if len(rewards_q_window)>window_size:
                rewards_q_window.pop(0)
            if len(successes_q_window)>window_size:
                successes_q_window.pop(0)

            episodes_q_history.append(episode_q)
            avg_rewards_q_history.append(np.mean(rewards_q_window))
            success_rate_q_history.append(np.mean(successes_q_window))

            agent_in_use.decay_epsilon()

        else:
            total_reward=0
            steps=0
            state=env.get_state()
            done=False
            while not done and not env.stop_training:
                old_pos=list(env.agent_pos)
                action=agent_in_use.select_action(state)
                next_state,reward,done=env.step(action,old_pos=old_pos)
                agent_in_use.store_experience(state,action,reward,next_state,done)
                agent_in_use.update()
                total_reward+=reward
                state=next_state
                steps+=1

                if steps>MAX_STEPS_PER_EPISODE:
                    done=True

                env.handle_events()
                env.render(
                    episode_dqn,
                    agent_name,
                    np.mean(rewards_q_window) if rewards_q_window else 0,
                    np.mean(successes_q_window) if successes_q_window else 0,
                    np.mean(rewards_dqn_window) if rewards_dqn_window else 0,
                    np.mean(successes_dqn_window) if successes_dqn_window else 0
                )
                env.clock.tick(FPS)

            episode_dqn+=1
            rewards_dqn_window.append(total_reward)
            successes_dqn_window.append(1 if total_reward>0 else 0)
            if len(rewards_dqn_window)>window_size:
                rewards_dqn_window.pop(0)
            if len(successes_dqn_window)>window_size:
                successes_dqn_window.pop(0)

            episodes_dqn_history.append(episode_dqn)
            avg_rewards_dqn_history.append(np.mean(rewards_dqn_window))
            success_rate_dqn_history.append(np.mean(successes_dqn_window))

            agent_in_use.decay_epsilon()

    # Fine => Salvataggio e chiusura
    agent_in_use.save_model()
    env.close()

    # =========================================================================
    # GRAFICI FINALI
    # =========================================================================

    # Q-Learning
    if len(episodes_q_history)>0:
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(episodes_q_history,avg_rewards_q_history,
                 label='Q-Learning Avg Reward',color='blue')
        plt.xlabel('Episodi (Q-Learning)')
        plt.ylabel('Reward Medio')
        plt.title('Reward Medio - Q-Learning')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(episodes_q_history,[sr*100 for sr in success_rate_q_history],
                 label='Q-Learning Success Rate',color='green')
        plt.xlabel('Episodi (Q-Learning)')
        plt.ylabel('Success Rate (%)')
        plt.title('Tasso di Successo - Q-Learning')
        plt.legend()

        plt.tight_layout()
        plt.show()

    # DQN
    if len(episodes_dqn_history)>0:
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(episodes_dqn_history,avg_rewards_dqn_history,
                 label='DQN Avg Reward',color='orange')
        plt.xlabel('Episodi (DQN)')
        plt.ylabel('Reward Medio')
        plt.title('Reward Medio - Dueling Double DQN')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(episodes_dqn_history,[sr*100 for sr in success_rate_dqn_history],
                 label='DQN Success Rate',color='red')
        plt.xlabel('Episodi (DQN)')
        plt.ylabel('Success Rate (%)')
        plt.title('Tasso di Successo - Dueling Double DQN')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    main()
