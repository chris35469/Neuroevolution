import pygame
import random
import torch
import time

# Constants
WIDTH = 800
HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (135, 206, 235)  # Sky blue
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

# Game constants
GRAVITY = 0.5
JUMP_STRENGTH = -10
PIPE_WIDTH = 80
PIPE_GAP = 200
PIPE_SPEED = 5
PIPE_SPAWN_RATE = 1500  # milliseconds

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.radius = 20
        self.color = YELLOW
        # Track how long the bird has been alive (frames)
        self.birth = time.time()
        self.time_alive = 0
        self.is_gravity = False
        
    def get_age(self):
        return time.time() - self.birth
    
    def set_gravity(self, is_gravity):
        self.is_gravity = is_gravity

    def die(self):
        self.time_alive =  time.time() - self.birth

    def get_time_alive(self):
        return self.time_alive
    
    def jump(self):
        self.velocity = JUMP_STRENGTH
        
    def update(self):
        if self.is_gravity:
            self.velocity += GRAVITY
            self.y += self.velocity
        
        # Prevent bird from going off screen
        if self.y < 0:
            self.y = 0
            self.velocity = 0
        if self.y > HEIGHT:
            self.y = HEIGHT
            self.velocity = 0
            
    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        # Draw a simple eye
        pygame.draw.circle(screen, BLACK, (int(self.x + 8), int(self.y - 5)), 5)
        
    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH
        self.gap = PIPE_GAP
        self.gap_y = random.randint(150, HEIGHT - 150)
        self.passed = False
        
    def update(self):
        self.x -= PIPE_SPEED
        
    def draw(self, screen):
        # Top pipe
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - self.gap // 2)
        pygame.draw.rect(screen, GREEN, top_rect)
        pygame.draw.rect(screen, BLACK, top_rect, 3)
        
        # Bottom pipe
        bottom_y = self.gap_y + self.gap // 2
        bottom_rect = pygame.Rect(self.x, bottom_y, self.width, HEIGHT - bottom_y)
        pygame.draw.rect(screen, GREEN, bottom_rect)
        pygame.draw.rect(screen, BLACK, bottom_rect, 3)
        
    def get_rects(self):
        top_rect = pygame.Rect(self.x, 0, self.width, self.gap_y - self.gap // 2)
        bottom_y = self.gap_y + self.gap // 2
        bottom_rect = pygame.Rect(self.x, bottom_y, self.width, HEIGHT - bottom_y)
        return [top_rect, bottom_rect]
    
    def is_off_screen(self):
        return self.x + self.width < 0

def check_collision(bird, pipes):
    bird_rect = bird.get_rect()
    #print(bird_rect.x, bird_rect.y)
    
    # Check collision with top/bottom of screen
    if bird.y - bird.radius <= 0 or bird.y + bird.radius >= HEIGHT:
        return True

    # if len(pipes) > 0:
    #     dx, dy = get_pipe_bird_distance(bird, pipes[0])
    #     print(dx, dy)
    
    # Check collision with pipes
    for pipe in pipes:
        for pipe_rect in pipe.get_rects():
            if bird_rect.colliderect(pipe_rect):
                return True
    
    return False

def draw_text(screen, text, size, x, y, color=BLACK):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    screen.blit(text_surface, text_rect)


def get_pipe_bird_distance(bird, pipe):
    """
    Calculate the horizontal (x) and vertical (y) distance between the bird and the nearest gap in the pipe.

    Returns:
        (dx, dy) where:
            dx: distance from bird's center to the front edge of the pipe (pipe.x)
            dy: distance from bird's center to the center of the pipe's gap
    """
    dx = pipe.x - bird.x
    gap_center_y = pipe.gap_y
    dy = gap_center_y - bird.y
    return float(dx), float(dy)


def start_game(model):
    # Initialize Pygame
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Bird")
    clock = pygame.time.Clock()
    
    # Game state
    game_over = False
    score = 0
    last_pipe_time = 0
    
    # Initialize game objects
    bird = Bird(150, HEIGHT // 2)
    pipes = []
    
    running = True
    while running:
        clock.tick(FPS)
        current_time = pygame.time.get_ticks()


        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game_over:
                        # Restart game
                        game_over = False
                        score = 0
                        bird = Bird(150, HEIGHT // 2)
                        pipes = []
                        last_pipe_time = current_time
                    else:
                        bird.jump()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if game_over:
                        # Restart game
                        game_over = False
                        score = 0
                        bird = Bird(150, HEIGHT // 2)
                        pipes = []
                        last_pipe_time = current_time
                    else:
                        bird.jump()
        
        if not game_over:
            # Update bird
            bird.update()
            
            # Spawn new pipes
            if current_time - last_pipe_time > PIPE_SPAWN_RATE:
                pipes.append(Pipe(WIDTH))
                last_pipe_time = current_time
            
            # Update pipes
            for pipe in pipes[:]:
                pipe.update()
                
                # Check if bird passed the pipe
                if not pipe.passed and pipe.x + pipe.width < bird.x:
                    pipe.passed = True
                    score += 1
                
                # Remove off-screen pipes
                if pipe.is_off_screen():
                    pipes.remove(pipe)

            if len(pipes) > 0:
                if bird.is_gravity == False:
                    bird.set_gravity(True)

                dx, dy = get_pipe_bird_distance(bird, pipes[0])

                with torch.no_grad():
                    X = torch.tensor([[dx, dy, bird.y]], dtype=torch.float32)
                    y = model(X)
                    predictions = torch.argmax(y, dim=1)
                    if predictions.numpy()[0] == 0:
                        print("Jumping")
                        bird.jump()
            else:
                if random.random() < 0.1:
                    bird.jump()
             
                    # print(y)
                    # print(predictions.numpy()[0])
            # Check collisions
            if check_collision(bird, pipes):
                game_over = True
                bird.die()
                print(f"Bird lived for {bird.get_time_alive()} seconds")
        
        # Drawing
        screen.fill(BLUE)  # Sky background
        
        if not game_over:
            # Draw pipes
            for pipe in pipes:
                pipe.draw(screen)
            
            # Draw bird
            bird.draw(screen)
        else:
            # Draw game over screen
            draw_text(screen, "GAME OVER", 72, WIDTH // 2, HEIGHT // 2 - 50, RED)
            draw_text(screen, f"Score: {score}", 48, WIDTH // 2, HEIGHT // 2 + 20)
            draw_text(screen, "Press SPACE or CLICK to restart", 36, WIDTH // 2, HEIGHT // 2 + 80)
            #pygame.quit()
            break
            
        
        # Draw score
        if not game_over:
            draw_text(screen, f"Score: {score}", 48, WIDTH // 2, 50, WHITE)
        
        # Update display
        pygame.display.flip()
    
    pygame.quit()
    return bird.get_time_alive()

# if __name__ == "__main__":
#     start_game()
