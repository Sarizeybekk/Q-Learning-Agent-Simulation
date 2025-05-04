import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, \
    QStatusBar
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt5.QtCore import Qt, QTimer

# Ortam Ayarları
grid_size = 5
actions = ['up', 'down', 'left', 'right', 'up-right', 'up-left', 'down-right', 'down-left']
reward_goal = 1
reward_obstacle = -1
reward_step = -0.01
start_state = (0, 0)
goal_state = (4, 4)
obstacles = [(3, 3)]
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Hiperparametreler
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 500


# Hareket Fonksiyonu
def take_action(state, action):
    x, y = state
    if action == 'up':
        x = max(0, x - 1)
    elif action == 'down':
        x = min(grid_size - 1, x + 1)
    elif action == 'left':
        y = max(0, y - 1)
    elif action == 'right':
        y = min(grid_size - 1, y + 1)
    elif action == 'up-right':
        x = max(0, x - 1)
        y = min(grid_size - 1, y + 1)
    elif action == 'up-left':
        x = max(0, x - 1)
        y = max(0, y - 1)
    elif action == 'down-right':
        x = min(grid_size - 1, x + 1)
        y = min(grid_size - 1, y + 1)
    elif action == 'down-left':
        x = min(grid_size - 1, x + 1)
        y = max(0, y - 1)
    return (x, y)


# Eğitim
rewards_per_episode = []
for episode in range(num_episodes):
    state = start_state
    total_reward = 0
    for step in range(100):
        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, len(actions) - 1)
        else:
            action_index = np.argmax(q_table[state[0], state[1]])
        action = actions[action_index]
        new_state = take_action(state, action)
        if new_state in obstacles:
            reward = reward_obstacle
        elif new_state == goal_state:
            reward = reward_goal
        else:
            reward = reward_step
        old_q_value = q_table[state[0], state[1], action_index]
        next_max = np.max(q_table[new_state[0], new_state[1]])
        new_q_value = old_q_value + learning_rate * (reward + discount_factor * next_max - old_q_value)
        q_table[state[0], state[1], action_index] = new_q_value
        state = new_state
        total_reward += reward
        if state == goal_state:
            break
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    rewards_per_episode.append(total_reward)

# Eğitim Grafiği
plt.figure(figsize=(10, 6))
plt.plot(rewards_per_episode, color='#2c3e50')
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Total Reward', fontsize=12)
plt.title('Training Progress - Total Reward per Episode', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('training_progress.png')
plt.close()


# Özel Grid Widget
class GridWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cell_size = 80
        self.state = start_state
        self.setMinimumSize(grid_size * self.cell_size, grid_size * self.cell_size)

    def set_state(self, state):
        self.state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Dinamik hücre boyutu
        width = self.width()
        height = self.height()
        self.cell_size = min(width, height) // grid_size

        # Izgara çizimi
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * self.cell_size
                y = i * self.cell_size
                painter.setPen(QPen(QColor(149, 165, 166), 2))
                painter.setBrush(QBrush(QColor(236, 240, 241)))
                painter.drawRect(x, y, self.cell_size, self.cell_size)

                # Ajan, hedef ve engel
                if (i, j) == self.state:
                    painter.setBrush(QBrush(QColor(52, 152, 219)))  # Mavi
                    painter.drawEllipse(x + 10, y + 10, self.cell_size - 20, self.cell_size - 20)
                elif (i, j) == goal_state:
                    painter.setBrush(QBrush(QColor(46, 204, 113)))  # Yeşil
                    painter.drawRect(x + 5, y + 5, self.cell_size - 10, self.cell_size - 10)
                elif (i, j) in obstacles:
                    painter.setBrush(QBrush(QColor(231, 76, 60)))  # Kırmızı
                    painter.drawRect(x + 5, y + 5, self.cell_size - 10, self.cell_size - 10)


# PyQt5 Arayüz
class GridWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Learning Agent Simulation")
        self.base_width = grid_size * 80 + 20
        self.base_height = grid_size * 80 + 150
        self.setMinimumSize(self.base_width, self.base_height)

        # Ana widget ve layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Grid widget
        self.grid_widget = GridWidget()
        self.main_layout.addWidget(self.grid_widget)

        # Kontrol paneli
        self.control_panel = QWidget()
        self.control_layout = QHBoxLayout(self.control_panel)

        self.start_button = QPushButton("Start")
        self.start_button.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px; border-radius: 4px;")
        self.start_button.clicked.connect(self.start_simulation)
        self.control_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setStyleSheet("background-color: #e67e22; color: white; padding: 8px; border-radius: 4px;")
        self.pause_button.clicked.connect(self.pause_simulation)
        self.control_layout.addWidget(self.pause_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px; border-radius: 4px;")
        self.reset_button.clicked.connect(self.reset_simulation)
        self.control_layout.addWidget(self.reset_button)

        self.main_layout.addWidget(self.control_panel)

        # Bilgi paneli
        self.info_panel = QWidget()
        self.info_layout = QHBoxLayout(self.info_panel)
        self.step_label = QLabel("Steps: 0")
        self.step_label.setFont(QFont("Arial", 12))
        self.reward_label = QLabel("Reward: 0.00")
        self.reward_label.setFont(QFont("Arial", 12))
        self.info_layout.addWidget(self.step_label)
        self.info_layout.addWidget(self.reward_label)
        self.main_layout.addWidget(self.info_panel)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready to start simulation")

        # Simülasyon değişkenleri
        self.state = start_state
        self.steps = 0
        self.total_reward = 0
        self.is_running = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_simulation)

        # Stil
        self.setStyleSheet("""
            QMainWindow { background-color: #ecf0f1; }
            QLabel { color: #2c3e50; }
        """)

    def start_simulation(self):
        if self.is_running:
            return
        self.is_running = True
        self.timer.start(300)
        self.statusBar.showMessage("Simulation running...")
        self.start_button.setText("Resume")
        self.start_button.setStyleSheet("background-color: #3498db; color: white; padding: 8px; border-radius: 4px;")

    def pause_simulation(self):
        if not self.is_running:
            return
        self.is_running = False
        self.timer.stop()
        self.statusBar.showMessage("Simulation paused")
        self.start_button.setText("Resume")

    def reset_simulation(self):
        self.state = start_state
        self.steps = 0
        self.total_reward = 0
        self.is_running = False
        self.timer.stop()
        self.step_label.setText("Steps: 0")
        self.reward_label.setText("Reward: 0.00")
        self.statusBar.showMessage("Simulation reset")
        self.start_button.setText("Start")
        self.start_button.setStyleSheet("background-color: #2ecc71; color: white; padding: 8px; border-radius: 4px;")
        self.grid_widget.set_state(self.state)

    def update_simulation(self):
        if self.state != goal_state and self.steps < 50:
            action_index = np.argmax(q_table[self.state[0], self.state[1]])
            action = actions[action_index]
            new_state = take_action(self.state, action)

            # Ödül hesapla
            if new_state in obstacles:
                reward = reward_obstacle
            elif new_state == goal_state:
                reward = reward_goal
            else:
                reward = reward_step

            self.state = new_state
            self.steps += 1
            self.total_reward += reward

            self.step_label.setText(f"Steps: {self.steps}")
            self.reward_label.setText(f"Reward: {self.total_reward:.2f}")
            self.grid_widget.set_state(self.state)

            if self.state == goal_state:
                self.timer.stop()
                self.is_running = False
                self.statusBar.showMessage(f"Goal reached in {self.steps} steps!")
                self.start_button.setText("Start")
                self.start_button.setStyleSheet(
                    "background-color: #2ecc71; color: white; padding: 8px; border-radius: 4px;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GridWindow()
    window.show()
    sys.exit(app.exec_())