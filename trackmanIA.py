import math
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Paramètres du jeu
WIDTH, HEIGHT = 1920, 1080
CAR_SIZE = (60, 60)
BORDER_COLOR = (255, 255, 255, 255)
REP = 10000
MAP = "map.png"

# Hyperparamètres du Deep Q-Learning
GAMMA = 0.95
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.target_model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.output_dim = output_dim

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

class Car:
    def __init__(self):
        super.__init__()
        self.position = [830, 920]
        self.angle = 0
        self.alive = True
        self.radar = []
        self.speed = 10
        self.car_image = pygame.image.load("car.png")
        self.car_image = pygame.transform.scale(self.car_image, (30, 30))

    def update_pos(self, input):
        if input == 1:
            self.angle += 10
        elif input == 0:
            self.angle -= 10
        
        self.position[0] += math.cos(math.radians(self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(self.angle)) * self.speed

    def collision(self, game_map):
        car_rect = self.car_image.get_rect(center=(self.position[0], self.position[1]))
        for x in range(car_rect.left, car_rect.right):
            for y in range(car_rect.top, car_rect.bottom):
                if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                    if game_map.get_at((x, y)) == BORDER_COLOR:
                        self.alive = False
                        return True
        return False

    def check_radar(self, degree, game_map):
        length = 0
        x, y = int(self.position[0]), int(self.position[1])
        while length < 300:
            x = int(self.position[0] + math.cos(math.radians(self.angle + degree)) * length)
            y = int(self.position[1] + math.sin(math.radians(self.angle + degree)) * length)
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if game_map.get_at((x, y)) == BORDER_COLOR:
                    break
            length += 1
        return length / 300.0

    def state(self, game_map):
        self.radar.clear()
        for d in range(-90, 120, 45):
            self.radar.append(self.check_radar(d, game_map))
        return np.array(self.radar)

pygame.init()
écran = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
agent = DQNAgent(input_dim=5, output_dim=3)

game_map = pygame.image.load(MAP)

for episode in range(REP):
    car = Car()
    state = car.state(game_map)
    final_reward = 0

    while car.alive:
        action = agent.select_action(state)
        car.update_pos(action)
        done = car.collision(game_map)
        next_state = car.state(game_map)

        reward = -10 if done else 1
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        final_reward += reward

        écran.fill((0, 0, 0))
        écran.blit(game_map, (0, 0))
        écran.blit(car.car_image, car.car_image.get_rect(center=(int(car.position[0]), int(car.position[1]))))
        pygame.display.flip()
        clock.tick(60)

    agent.update_epsilon()
    print(f"Épisode {episode}, Score: {final_reward}, Epsilon: {agent.epsilon:.2f}")

pygame.quit()