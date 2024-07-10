import pygame
import torch
import torch.nn.functional as F
from torch import nn

from dino_game.init_dino import DinoGame


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(2, 1)
        self.flatten = nn.Flatten()

        dummy_input = torch.randn(1, 3, 34, 34)
        conv_out_size = self._get_conv_out(dummy_input)

        self.fc1 = nn.Linear(conv_out_size, 16)
        self.output_layer = nn.Linear(16, out_features=n_actions)

    def _get_conv_out(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        return int(torch.prod(torch.tensor(x.shape[1:])))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.output_layer(x)
        return x

def run_game():
    pygame.init()
    dino = DinoGame()
    dqn = DQN(n_actions=3)  # Assuming 3 actions: [no action, jump, duck]
    dqn.eval()  # Set the model to evaluation mode (no gradient computation)

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

            # Convert state to tensor and adjust dimensions
            # state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                q_values = dqn(state)
                action = torch.argmax(q_values).item()
                print("Action :" ,action)  # Check the action chosen by the agent

            # Take action and get next state, reward, done
            next_state, reward, done = dino.step(action)

            # Render the game
            dino.render()

            # Update state for next iteration
            state = next_state

            clock.tick(30)  # Limit to 30 FPS

        if not running:
            break

    pygame.quit()

if __name__ == "__main__":
    run_game()