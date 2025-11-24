from Model import Individual
from utils import load_csv, save_to_csv
import random
import os
from run_all import (GENERATIONS, POPULATION_SIZE, BEST_INDIVIDUALS_SIZE,
                     CROSSOVER_RATE, MUTATION_RATE, MUTATION_STRENGTH,
                     SAVE_MODELS, SAVE_CSV_RESULTS)

BASE_DIRECTORY = 'notrain_ycommand'
RUN_TYPE = 'y_command'

# Import shared configuration from run_all.py
generations = GENERATIONS
population_size = POPULATION_SIZE
best_individuals_size = BEST_INDIVIDUALS_SIZE
crossover_rate = CROSSOVER_RATE
mutation_rate = MUTATION_RATE
mutation_strength = MUTATION_STRENGTH

# Load data (CSV conversion is handled by run_all.py)
X = load_csv('data/csv_output/matrix.csv')
y_command = load_csv('data/csv_output/command.csv')
y_continuous = load_csv('data/csv_output/continuous_command.csv')

# Select target based on RUN_TYPE
target_map = {
    'y_command': y_command,
    'y_continuous': y_continuous
}
y_target = target_map[RUN_TYPE]


# ---------------------------------------------------------------------------------------------------------------------------------
# Genetic Algorithm Implementation ------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Loop through generations
individuals = []

for generation in range(generations):

    if generation == 0:
        # INITIAL POPULATION: Create random individuals
        # print("Creating initial random population...")
        individuals = []
        for i in range(population_size):
            individual_id = f"{generation}_{i}"
            individual = Individual(input_features=4096, h1=512, h2=256, h3=128, output_features=3, individual_id=individual_id, verbose=False)
            individual.generation = generation
            fitness = individual.evaluate_fitness(X, y_target)
            individuals.append(individual)
            if SAVE_MODELS:
                individual.save(f'{BASE_DIRECTORY}/models/{RUN_TYPE}/model_{individual.individual_id}')
    else:
        # SUBSEQUENT GENERATIONS: Evolve from previous generation
        # print("Evolving population...")
        
        # 1. Sort by fitness (best first) - using Individual's __lt__ method
        individuals.sort()  # Best individuals first (highest fitness)
        
        # 2. SELECT: Keep top performers (elites)
        elites = individuals[:best_individuals_size]
        # print(f"  Keeping top {best_individuals_size} elites:")
        # for i, elite in enumerate(elites):
        #     print(f"    Elite {i+1}: {elite}")
        
        # 3. CROSSOVER: Create offspring from elites
        offspring = []
        offspring_count = population_size - best_individuals_size
        # print(f"  Creating {offspring_count} offspring via crossover...")
        
        for i in range(offspring_count):
            # Select two parents randomly from elites (can be same parent twice)
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            
            # Create offspring via crossover
            child = parent1.crossover(parent2, crossover_rate=crossover_rate)
            child.individual_id = f"{generation}_{i}"
            child.generation = generation
            
            # Apply mutation to introduce diversity
            child.mutate(mutation_rate=mutation_rate, mutation_strength=mutation_strength)
            
            offspring.append(child)
            # print(f"    Offspring {i+1}: Parents {parent1.individual_id} × {parent2.individual_id} (mutated)")
        
        # 4. REPLACE: Form next generation (elites + offspring)
        individuals = elites + offspring
        # print(f"  Next generation: {len(elites)} elites + {len(offspring)} offspring = {len(individuals)} individuals")
    
    # Evaluate fitness for all individuals (offspring need evaluation)
    # print("Evaluating fitness...")
    fitnesses = []
    for i, individual in enumerate(individuals):
        if individual.fitness is None:  # Only evaluate if not already evaluated
            fitness = individual.evaluate_fitness(X, y_target)
        else:
            fitness = individual.fitness
        fitnesses.append(fitness)
        if SAVE_MODELS:
            individual.save(f'{BASE_DIRECTORY}/models/{RUN_TYPE}/model_{individual.individual_id}')
    
    # Save results (optional)
    if SAVE_CSV_RESULTS:
        save_to_csv(fitnesses, f'{BASE_DIRECTORY}/results/fitnesses_generation_{generation}.csv')
        save_to_csv(individuals, f'{BASE_DIRECTORY}/results/individuals_generation_{generation}.csv')
    
    # Display generation statistics
    best_fitness = max(fitnesses)
    worst_fitness = min(fitnesses)
    avg_fitness = sum(fitnesses)/len(fitnesses)
    
    print(f"\nGeneration {generation} Statistics:")
    print(f"  Best fitness: {best_fitness:.6f}")
    print(f"  Worst fitness: {worst_fitness:.6f}")
    print(f"  Average fitness: {avg_fitness:.6f}")

# Find the best individual from the final generation
individuals.sort()  # Sort by fitness (best first)
best_individual = individuals[0]

# Create best_model directory path
best_model_path = f'{BASE_DIRECTORY}/best_model/best_model_{RUN_TYPE}'

# Save the best model
best_individual.save(best_model_path)
print(f"\n✓ Best model saved to: {best_model_path}.pth")
print(f"  Best fitness: {best_individual.fitness:.6f}")
print(f"  Individual ID: {best_individual.individual_id}")
print(f"  Generation: {best_individual.generation}")

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------