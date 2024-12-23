import numpy as np
import random
import pygame
from enum import Enum
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


# Определение направлений движения
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Размер блока и скорость игры
BLOCK_SIZE = 20
SPEED = 40

# Цвета (RGB)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


# Класс игры
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Инициализация отображения
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Инициализация состояния игры
        self.direction = Direction.RIGHT
        self.head = [self.w / 2, self.h / 2]
        self.snake = [self.head.copy(),
                      [self.head[0] - BLOCK_SIZE, self.head[1]],
                      [self.head[0] - (2 * BLOCK_SIZE), self.head[1]]]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Движение
        self._move(action)  # Обновление головы
        self.snake.insert(0, self.head.copy())

        # Проверка на столкновение
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Поедание еды
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # Обновление UI и часы
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Столкновение с границами
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] > self.h - BLOCK_SIZE or pt[1] < 0:
            return True
        # Столкновение с собой
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt[0], pt[1], BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

    def _move(self, action):
        # [1, 0, 0] - прямо
        # [0, 1, 0] - вправо
        # [0, 0, 1] - влево

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Без изменения
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Поворот вправо
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Поворот влево

        self.direction = new_dir

        x = self.head[0]
        y = self.head[1]
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = [x, y]


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Параметр для ε-greedy
        self.gamma = 0.9  # Коэффициент дисконтирования
        self.memory = deque(maxlen=100_000)
        self.model = Linear_QNet(11, 256, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def get_state(self, game):
        head = game.snake[0]
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Опасность прямо
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Опасность вправо
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # Опасность влево
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # Направление движения
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Положение еды относительно головы
            game.food[0] < game.head[0],  # Еда слева
            game.food[0] > game.head[0],  # Еда справа
            game.food[1] < game.head[1],  # Еда сверху
            game.food[1] > game.head[1]  # Еда снизу
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.train_step([state], [action], [reward], [next_state], [done])

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)

        # Предсказание текущего Q-значения
        pred = self.model(states)

        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(self.model(next_states[idx]))
            target[idx][torch.argmax(actions[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


def train():
    pygame.init()
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # Получение текущего состояния
        state_old = agent.get_state(game)

        # Получение действия
        final_move = agent.get_action(state_old)

        # Выполнение действия и получение нового состояния
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Обучение короткой памяти
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Запоминание
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Обучение долгой памяти
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            print('Игра', agent.n_games, 'Счет', score)


if __name__ == '__main__':
    train()