from environment import MultiCarRacing
import numpy as np


env = MultiCarRacing(n_cars=4, grid_size=30, track_width=5, num_checkpoints=12, render_mode=None)
env.reset()

while True:
    actions = {agent_id: np.random.randint(0,5) for agent_id in range(4)}  # Your action selection logic here
    obs, rewards, dones, info = env.step(actions)
    env.render()
    print(rewards)
    if any(dones.values()):
        print(dones)
        break