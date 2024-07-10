import io
import threading

import pygame
from torch import nn

from dino_game.utils.init_dino import DinoGame

class DQN(nn.Module):
    def __init__(self,n_observations,n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3,1,kernel_size=3,stride=2)
        self.pool1 = nn.MaxPool2d(2,1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1,16)
        self.act1 = nn.ReLU()
        self.fc1 = nn.Linear(out_features=n_actions)

stream = io.BytesIO()

def run_game():
    pygame.init()
    dino = DinoGame()
    clock = pygame.time.Clock()
    running = True

    while running:
        state = dino.reset()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True

            # Here you would normally get an action from your agent
            action = 0  # placeholder, replace with actual action selection
            state, reward, done = dino.step(action)
            print(state.shape)
            dino.render()

            clock.tick(30)  # limit to 30 FPS

        if not running:
            break

    pygame.quit()

if __name__ == "__main__":
    run_game()