import math
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from agent import Car
import pygame

GRID_SIZE = 30
CELL_SIZE = 30  # Each cell's width and height in pixels
SCREEN_SIZE = GRID_SIZE * CELL_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (124, 252, 0)  # Starting line color
BLUE = (0, 0, 255)     # Finish line color
GRAY = (128, 128, 128)  # Track color
CAR_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Car colors (extendable)


def create_track_and_checkpoints(grid_size: int, track_width: int, num_checkpoints: int):
    track = set()
    checkpoints = []

    # Define the center and radius for the circular track
    center_x, center_y = grid_size // 2, grid_size // 2
    outer_radius = grid_size // 2 - 1
    inner_radius = outer_radius - track_width

    # Create track
    for x in range(grid_size):
        for y in range(grid_size):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if inner_radius < distance < outer_radius:
                track.add((x, y))

    for angle in range(0, 360, 360 // num_checkpoints):
        checkpoint = []
        rad_angle = math.radians(angle)
        
        # Calculate the center point of this checkpoint strip
        center_checkpoint_x = center_x + (inner_radius + track_width / 2) * math.sin(rad_angle)
        center_checkpoint_y = center_y - (inner_radius + track_width / 2) * math.cos(rad_angle)
        
        # Calculate the perpendicular angle for the checkpoint strip (90 degrees offset)
        perp_angle = rad_angle + math.pi / 2
        
        # Generate the entire track width for each checkpoint
        for w in range(-track_width // 2, track_width // 2 + 1):
            # Adjust position along the track
            x = int(center_checkpoint_x + w * math.cos(perp_angle))
            y = int(center_checkpoint_y + w * math.sin(perp_angle))
            
            # Check if the coordinates are within grid bounds
            if 0 <= x < grid_size and 0 <= y < grid_size:
                checkpoint.append((x, y))
        
        # If valid checkpoint generated, add it to checkpoints list
        if checkpoint:
            checkpoints.append(checkpoint)

    return track, checkpoints



class MultiCarRacing(MultiAgentEnv):
    def __init__(self, n_cars: int, grid_size: int, track_width: int, render_mode=None):
        super().__init__()
        self.n_cars = n_cars
        self.grid_rows = grid_size
        self.grid_columns = grid_size
        self.agents = {agent_id: Car(agent_id) for agent_id in range(n_cars)}

        self.action_space = {agent_id: [0, 1, 2, 3, 4] for agent_id in range(n_cars)}
        self.observation_space = {
            agent_id :  np.zeros((grid_size, grid_size)) for agent_id in range(n_cars)
        }

        self.track, self.checkpoints = create_track_and_checkpoints(grid_size, track_width, 12)

        # Create start line at the top of the circle
        self.start_line = [(grid_size // 2, w + 2) for w in range(track_width)]

        # Pygame Setup for rendering
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            pygame.display.set_caption("Multi-Agent Racetrack Environment")
            self.clock = pygame.time.Clock()


    def reset(self):
        self.rewards = {agent_id: 0 for agent_id in range(self.n_cars)}
        self.dones = {agent_id: False for agent_id in range(self.n_cars)}

        # reset car positions
        for agent_id, agent in self.agents.items():
            position = self.start_line[agent_id % len(self.start_line)]
            agent.reset(position, self.observation_space)

        return {agent_id: agent.observation for agent_id, agent in self.agents.items()}
    
    def step(self, action_dict):

        observations = {}
        rewards = {}
        infos = {}
        dones = {}

        intended_position = {agent_id: agent.position for agent_id, agent in self.agents.items()}

        # Position Update
        for agent_id, agent in self.agents.items():
            if self.dones[agent_id]:
                pass # Add dead step here
            else:
                action = action_dict[agent_id]
                position = agent.position
                if agent.collision_counter > 0:
                    agent.collision_counter -= 1
                else:
                    # Handle movement based on action
                    if action == 0:  # Move left
                        new_pos = (position[0] - 1, position[1])
                    elif action == 1:  # Move right
                        new_pos = (position[0] + 1, position[1])
                    elif action == 2:  # Move forward
                        new_pos = (position[0], position[1] + 1)
                    elif action == 3:  # Move back
                        new_pos = (position[0], position[1] - 1)
                    else:  # Stay in place
                        new_pos = position
                    
                    # Check if the new position is within the track boundaries
                    intended_position[agent_id] = new_pos if new_pos in self.track else position
                
                # Check collision with other cars
                for other_agent_id, other_agent in self.agents.items():
                    if agent_id != other_agent_id and intended_position[agent_id] == intended_position[other_agent_id]:
                        agent.collision_counter = 2
                        intended_position[agent_id] = position
                        other_agent.collision_counter = 2
                
            
            # Reward update
            for agent_id, agent in self.agents.items():
                agent.position = intended_position[agent_id]

                if intended_position in self.checkpoints[agent.checkpoint_counters]:
                    agent.reward += 5
                    agent.checkpoint_counters += 1

                    if agent.checkpoint_counters >= len(self.checkpoints):
                        agent.done = True
                agent.observation = self.get_observation(agent_id)
                self.dones[agent_id] = agent.done
            
        observations = {
            agent_id: agent.observation for agent_id, agent in self.agents.items()
        }

        rewards = {
            agent_id: agent.reward for agent_id, agent in self.agents.items()
        }
        
        return observations, rewards, self.dones, infos

    def render(self):
        if self.render_mode != "human":
            return

        # Fill screen with white background
        self.screen.fill(WHITE)
        
        # Draw track
        for x, y in self.track:
            pygame.draw.rect(
                self.screen,
                GRAY,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
        
        # Draw checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            # Use a gradient of colors from green to blue for checkpoints
            color = (
                int(124 * (1 - i/len(self.checkpoints))),  # R
                int(252 * (1 - i/len(self.checkpoints))),  # G
                int(255 * (i/len(self.checkpoints)))       # B
            )
            for x, y in checkpoint:
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1  # Draw just the border
                )
        
        # Draw start line in green
        for x, y in self.start_line:
            pygame.draw.rect(
                self.screen,
                GREEN,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                2  # Thicker border for start line
            )
        
        # Draw cars
        for agent_id, agent in self.agents.items():
            x, y = agent.position
            
            # Calculate car color based on team (every 2 cars share a color)
            color = CAR_COLORS[agent_id // 2 % len(CAR_COLORS)]
            
            # Draw car body
            car_rect = pygame.Rect(
                x * CELL_SIZE + 2,  # Add small padding
                y * CELL_SIZE + 2,
                CELL_SIZE - 4,
                CELL_SIZE - 4
            )
            pygame.draw.rect(self.screen, color, car_rect)
            
            # Draw checkpoint counter on car
            font = pygame.font.Font(None, 20)
            text = font.render(str(agent.checkpoint_counters), True, BLACK)
            text_rect = text.get_rect(center=(x * CELL_SIZE + CELL_SIZE//2,
                                            y * CELL_SIZE + CELL_SIZE//2))
            self.screen.blit(text, text_rect)
            
            # Draw collision indicator if car is in collision state
            if agent.collision_counter > 0:
                pygame.draw.circle(
                    self.screen,
                    (255, 0, 0),  # Red
                    (x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE - 5),
                    3
                )
        
        # Draw grid lines
        for i in range(GRID_SIZE + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                BLACK,
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, SCREEN_SIZE),
                1
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                BLACK,
                (0, i * CELL_SIZE),
                (SCREEN_SIZE, i * CELL_SIZE),
                1
            )
        
        # Add info text
        font = pygame.font.Font(None, 24)
        y_offset = 10
        for agent_id, agent in self.agents.items():
            text = f"Car {agent_id}: CP {agent.checkpoint_counters}/{len(self.checkpoints)}"
            text_surface = font.render(text, True, CAR_COLORS[agent_id // 2 % len(CAR_COLORS)])
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 25

        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    def close(self):
        if self.render_mode == "human":
            pygame.quit()

    def get_observation(self, agent_id):
        observation = np.zeros((self.grid_rows, self.grid_columns), dtype=int)

        # 1 for track
        for (x,y) in self.track:
            observation[x, y] = 1
        
        for other_agent_id, other_agent in self.agents.items():
            pos = other_agent.position
            # If self 2
            if other_agent_id == agent_id:
                observation[pos[0], pos[1]] = 2
            else:
                # If teammate 3
                if (agent_id // 2) == (other_agent_id // 2):
                    observation[pos[0], pos[1]] = 3
                # If enemy 4
                else:
                    observation[pos[0], pos[1]] = 4
        
        return observation
            

