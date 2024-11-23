import math
import numpy as np
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
    outer_radius = (grid_size // 2) - 2  # Slightly smaller to ensure visibility
    inner_radius = outer_radius - track_width

    # Create the track
    for y in range(grid_size):
        for x in range(grid_size):
            distance = math.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            if inner_radius <= distance < outer_radius:
                track.add((x, y))

    # Calculate angular positions of checkpoints
    angle_step = 360 / num_checkpoints
    for i in range(num_checkpoints):
        checkpoint = []
        angle = (i + 4) * angle_step  # Evenly distributed angles
        rad_angle = math.radians(angle)
        
        # Calculate the base point on the middle of the track
        mid_radius = (inner_radius + outer_radius) / 2
        base_x = center_x + mid_radius * math.cos(rad_angle)
        base_y = center_y + mid_radius * math.sin(rad_angle)
        
        # Decide if the checkpoint is horizontal or vertical
        is_horizontal = abs(math.sin(rad_angle)) < 0.707  # cos(45°) ≈ 0.707
        
        for w in range(-track_width - 1, track_width + 2):
            if is_horizontal:
                # Horizontal checkpoint (on sides)
                x = int(base_x + w)
                y = int(base_y)
            else:
                # Vertical checkpoint (top/bottom)
                x = int(base_x)
                y = int(base_y + w)
            
            if 0 <= x < grid_size and 0 <= y < grid_size and (x, y) in track:
                checkpoint.append((x, y))
        
        if checkpoint:
            checkpoints.append(checkpoint)

    # Create the start line
    start_line = []
    start_x = center_x  # Center position
    for y in range(grid_size // 2, grid_size):  # Only lower half of the grid
        if (start_x, y) in track:
            start_line.append((start_x, y))

    # Ensure the last checkpoint overlaps the start line
    if checkpoints:
        checkpoints[-1].extend(start_line)

    return track, checkpoints, start_line

class MultiCarRacing():
    def __init__(self, n_cars: int = 4, grid_size: int = 30, track_width: int = 5, num_checkpoints: int = 12, render_mode=None):
        super().__init__()
        self.n_cars = n_cars
        self.grid_rows = grid_size
        self.grid_columns = grid_size
        self.render_mode = render_mode
        
        # Create track first
        self.track, self.checkpoints, self.start_line = create_track_and_checkpoints(
            grid_size, track_width, num_checkpoints
        )
        
        # Initialize agents with starting positions
        self.agents = {}
        for agent_id in range(n_cars):
            start_pos = self.start_line[agent_id % len(self.start_line)]
            self.agents[agent_id] = Car(start_pos)

        self.action_space = {agent_id: [0, 1, 2, 3, 4] for agent_id in range(n_cars)}
        self.observation_space = {
            agent_id: np.zeros((grid_size, grid_size)) for agent_id in range(n_cars)
        }

        # Pygame Setup for rendering
        if render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
            pygame.display.set_caption("Multi-Agent Racetrack Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)


    def reset(self):
        # Reset each agent to its starting position
        for agent_id, agent in self.agents.items():
            start_pos = self.start_line[agent_id % len(self.start_line)]
            agent.position = start_pos
            agent.checkpoint_counters = 0
            agent.collision_counter = 0
            agent.done = False
        
        self.rewards = {agent_id: 0 for agent_id in self.agents}
        self.dones = {agent_id: False for agent_id in self.agents}
        
        return {agent_id: self.get_observation(agent_id) for agent_id in self.agents}
    
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

                if agent.position in self.checkpoints[agent.checkpoint_counters]:
                    agent.reward += 5
                    agent.checkpoint_counters += 1

                    if agent.checkpoint_counters >= len(self.checkpoints):
                        agent.done = True
                        agent.checkpoint_counters = 0
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

        # Clear screen with white background
        self.screen.fill(WHITE)
        
        # Draw track
        for x, y in self.track:
            pygame.draw.rect(
                self.screen,
                GRAY,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
        
        # Draw start line
        for x, y in self.start_line:
            pygame.draw.rect(
                self.screen,
                GREEN,
                (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            )
        
        # Draw checkpoints
        for i, checkpoint in enumerate(self.checkpoints):
            color = (
                int(124 + (131 * i/len(self.checkpoints))),  # R: 124 -> 255
                int(252 - (252 * i/len(self.checkpoints))),  # G: 252 -> 0
                int(0 + (255 * i/len(self.checkpoints)))     # B: 0 -> 255
            )
            for x, y in checkpoint:
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    2  # Border thickness
                )
            
            # Add checkpoint index number
            # Get the middle point of the checkpoint
            if checkpoint:  # Make sure checkpoint has points
                mid_point = checkpoint[len(checkpoint)//2]
                x, y = mid_point
                # Render the checkpoint number
                text = self.font.render(str(i), True, color)
                text_rect = text.get_rect(center=(
                    x * CELL_SIZE + CELL_SIZE/2,
                    y * CELL_SIZE + CELL_SIZE/2
                ))
                self.screen.blit(text, text_rect)
        
        # Draw cars
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'position'):  # Check if position exists
                x, y = agent.position
                color = CAR_COLORS[agent_id % len(CAR_COLORS)]
                
                # Draw car as a filled circle with a border
                center = (int(x * CELL_SIZE + CELL_SIZE/2), 
                         int(y * CELL_SIZE + CELL_SIZE/2))
                radius = int(CELL_SIZE/2) - 2
                
                # Draw filled circle
                pygame.draw.circle(self.screen, color, center, radius)
                # Draw border
                pygame.draw.circle(self.screen, BLACK, center, radius, 2)
                
                # Draw checkpoint counter
                if hasattr(agent, 'checkpoint_counters'):
                    text = self.font.render(str(agent.checkpoint_counters), True, BLACK)
                    text_rect = text.get_rect(center=center)
                    self.screen.blit(text, text_rect)
                
                # Draw collision indicator
                if hasattr(agent, 'collision_counter') and agent.collision_counter > 0:
                    pygame.draw.circle(
                        self.screen,
                        (255, 0, 0),  # Red
                        (center[0], center[1] - CELL_SIZE//2 - 5),
                        4
                    )
        
        # Draw grid lines
        for i in range(self.grid_rows + 1):
            # Vertical lines
            pygame.draw.line(
                self.screen,
                (200, 200, 200),  # Light gray
                (i * CELL_SIZE, 0),
                (i * CELL_SIZE, SCREEN_SIZE),
                1
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen,
                (200, 200, 200),  # Light gray
                (0, i * CELL_SIZE),
                (SCREEN_SIZE, i * CELL_SIZE),
                1
            )
        
        # Add info text
        y_offset = 10
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'checkpoint_counters'):
                text = f"Car {agent_id}: CP {agent.checkpoint_counters}/{len(self.checkpoints)}"
                text_surface = self.font.render(
                    text, True, CAR_COLORS[agent_id % len(CAR_COLORS)]
                )
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
                # If teammate 2
                if (agent_id // 2) == (other_agent_id // 2):
                    observation[pos[0], pos[1]] = 2
                # If enemy 3
                else:
                    observation[pos[0], pos[1]] = 3
        
        return observation
            

