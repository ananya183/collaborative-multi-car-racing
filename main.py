from environment import MultiCarRacing
import numpy as np


env = MultiCarRacing(n_cars=1, grid_size=30, track_width=5, num_checkpoints=12, render_mode="human")
env.reset()

print(env.checkpoints[0])

while True:
    actions = {agent_id: np.random.randint(0,5) for agent_id in range(4)}  # Your action selection logic here
    obs, rewards, dones, info = env.step(actions)
    print(f'checkpoint 0: {env.checkpoints[0]} agent_pos: {env.agents[0].position} agent checkpoint: {env.agents[0].checkpoint_counters}')
    env.render()
    
    if all(dones.values()):
        break