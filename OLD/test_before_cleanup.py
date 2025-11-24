from Model import CreateModel, ModelTrainer, Individual
from utils import load_json, json_to_csv, load_csv, loss

# Load data
# json_to_csv('data/dummy_data.json', 'data/csv_output')
X = load_csv('data/csv_output/matrix.csv')
y_command = load_csv('data/csv_output/command.csv')
y_continuous = load_csv('data/csv_output/continuous_command.csv')
# X, y_command, y_continuous = load_json('data/dummy_data.json')

# # Print the shape of the data
# print(f"X shape: {X.shape}")
# print(f"y_command shape: {y_command.shape}")
# print(f"y_continuous shape: {y_continuous.shape}")

# # Create the model
# model = CreateModel(input_features=4096, h1=512, h2=256, h3=128, output_features=3, verbose=False)

# # Create trainer and train
# trainer = ModelTrainer(model, learning_rate=0.001, verbose=False)
# trainer.train(X, y_command, epochs=100)

# ---------------------------------------------------------------------------------------------------------------------------------
# Genetic Algorithm Implementation (w/o training) PART 1 --------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
generations = 1000
population_size = 1000

models = {}
predictions = []
for i in range(10):
    individual = Individual(input_features=4096, h1=512, h2=256, h3=128, output_features=3, individual_id=i, verbose=False)
    fitness = individual.evaluate_fitness(X, y_command)
    models[i] = individual
    individual.save(save_path='models/genetic_algorithm_1', model_name=f'model_{i}')


# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------
# Genetic Algorithm Implementation (w/ training) PART 2 ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
# models_t = {}
# predictions_t = []
# for i in range(10):
#     model = CreateModel(input_features=4096, h1=512, h2=256, h3=128, output_features=3, verbose=False)
#     trainer = ModelTrainer(model, learning_rate=0.001, verbose=False)
#     trainer.train(X, y_command, epochs=100)
#     prediction = trainer.predict(X)
#     predictions_t.append(prediction)
#     models_t[i] = model
#     model.save_model(save_path='models/genetic_algorithm_training_1', model_name=f'model_{i}')

# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------

# Print predictions and models
# print(predictions)
# print(models)
# print(predictions_t)
# print(models_t)