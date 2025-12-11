import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import copy
from simplenet import SimpleNet
# Create a model class
class CreateModel(nn.Module):
    def __init__(self, input_features, h1, h2, h3, output_features, verbose=True):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, output_features)
        self.verbose = verbose
        
    def forward(self, x):
        if self.verbose:
            print(f"Input shape: {x.shape}")
        x = F.relu(self.fc1(x))
        if self.verbose:
            print(f"Output shape after fc1: {x.shape}")
        x = F.relu(self.fc2(x))
        if self.verbose:
            print(f"Output shape after fc2: {x.shape}")
        x = F.relu(self.fc3(x))
        if self.verbose:
            print(f"Output shape after fc3: {x.shape}")
        x = self.fc4(x)
        if self.verbose:
            print(f"Output shape after fc4: {x.shape}")
        return x
        
    def test(self, X):
        """
        Test the model by passing input data and returning output predictions.
        Useful for evolutionary algorithms where you want to evaluate models without training.
        
        Args:
            X: Input features tensor
            
        Returns:
            Output predictions tensor
        """
        self.eval()
        with torch.no_grad():
            predictions = self(X)
        return predictions
    
    def save_model(self, save_path, model_name):
        """
        Save the model to a specified location.
        
        Args:
            save_path: String path to the directory where the model should be saved (relative to current directory)
            model_name: String name for the model file (without extension)
            
        Returns:
            Full path where the model was saved
        """
        # Create the directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Construct the full file path
        full_path = os.path.join(save_path, f"{model_name}.pth")
        
        # Save the model's state_dict (recommended way to save PyTorch models)
        torch.save(self.state_dict(), full_path)
        
        return full_path


class ModelTrainer:
    """Class to handle model training."""
    
    def __init__(self, model, criterion=None, optimizer=None, learning_rate=0.001, verbose=True):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model to train
            criterion: Loss function (default: MSELoss)
            optimizer: Optimizer (default: Adam)
            learning_rate: Learning rate for optimizer (default: 0.001)
            verbose: If True, print loss for each epoch
        """
        self.model = model
        
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
            
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        self.verbose = verbose

    def train(self, X, y, epochs=100):
        """
        Train the model.
        
        Args:
            X: Input features tensor
            y: Target tensor
            epochs: Number of training epochs
        """
        self.model.train()
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.verbose:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        
        return loss.item()
    
    def predict(self, X):
        """
        Make predictions using the model.
        
        Args:
            X: Input features tensor
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate the model on given data and return the loss.
        
        Args:
            X: Input features tensor
            y: Target tensor
            
        Returns:
            Loss value
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
        return loss.item()


class Individual:
    """
    Wrapper class for CreateModel that holds genetic algorithm-related information.
    This class manages phenotypes, fitness, and genetic operations for evolutionary algorithms.
    """
    
    def __init__(self, model=None, input_features=4096, h1=512, h2=256, h3=128, output_features=3, 
                 verbose=False, individual_id=None):
        """
        Initialize an Individual for genetic algorithm.
        
        Args:
            model: Existing CreateModel instance (if None, creates a new one)
            input_features: Number of input neurons (used if model is None)
            h1: Size of first hidden layer (used if model is None)
            h2: Size of second hidden layer (used if model is None)
            h3: Size of third hidden layer (used if model is None)
            output_features: Number of output neurons (used if model is None)
            verbose: Verbose mode for model (default: False)
            individual_id: Unique identifier for this individual (optional)
        """
        # Create or use provided model
        if model is None:
            self.model = CreateModel(input_features, h1, h2, h3, output_features, verbose=verbose)
        else:
            self.model = model
        
        # Genetic algorithm attributes
        self.fitness = None  # Will hold fitness score (lower is better for loss-based fitness)
        self.individual_id = individual_id  # Unique identifier
        self.generation = 0  # Generation this individual was created in
        self.parent_ids = []  # IDs of parent individuals (for tracking lineage)
        
        # Metadata dictionary for storing additional information
        self.metadata = {
            'created_at': None,  # Can store timestamp
            'evaluated': False,  # Whether fitness has been evaluated
            'survived_generations': 0,  # How many generations this individual survived
            'mutations_applied': 0,  # Number of mutations applied
            'crossover_count': 0,  # Number of crossovers this individual participated in
        }
        
        # Store architecture info for cloning/crossover
        self.architecture = {
            'input_features': input_features if model is None else None,
            'h1': h1 if model is None else None,
            'h2': h2 if model is None else None,
            'h3': h3 if model is None else None,
            'output_features': output_features if model is None else None,
        }
    
    
    def set_fitness(self, fitness):
        self.fitness = fitness
        self.metadata['evaluated'] = True
    
    def evaluate_fitness(self, X, y, loss_fn=None):
        """
        Evaluate fitness of this individual.
        Returns negative loss so that higher fitness values indicate better performance.
        
        Args:
            X: Input features tensor
            y: Target tensor
            loss_fn: Loss function (if None, uses MSELoss)
            
        Returns:
            Fitness score (negative loss value, higher is better)
        """
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        predictions = self.model.test(X)
        loss_value = loss_fn(predictions, y).item()
        # Return negative loss so higher fitness = better performance
        self.fitness = -loss_value
        self.metadata['evaluated'] = True
        
        return self.fitness
    
    def clone(self, new_id=None):
        """
        Create a deep copy of this individual.
        
        Args:
            new_id: New ID for the cloned individual (if None, generates one)
            
        Returns:
            New Individual instance with copied model and metadata
        """
        # Deep copy the model's state_dict
        cloned_model = CreateModel(
            self.architecture['input_features'] or 4096,
            self.architecture['h1'] or 512,
            self.architecture['h2'] or 256,
            self.architecture['h3'] or 128,
            self.architecture['output_features'] or 3,
            verbose=self.model.verbose
        )
        cloned_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        
        # Create new Individual
        cloned = Individual(model=cloned_model, individual_id=new_id, verbose=self.model.verbose)
        
        # Copy metadata (but mark as new individual)
        cloned.metadata = copy.deepcopy(self.metadata)
        cloned.metadata['evaluated'] = False  # Reset evaluation status
        cloned.generation = self.generation
        cloned.parent_ids = [self.individual_id] if self.individual_id else []
        
        return cloned
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.1):
        """
        Apply mutation to the individual's model weights.
        TODO: Implement more sophisticated mutation strategies.
        
        Args:
            mutation_rate: Probability of mutating each weight (default: 0.1)
            mutation_strength: Strength of mutation (default: 0.1)
        """
        with torch.no_grad():
            for param in self.model.parameters():
                # Create random mask for mutation
                mask = torch.rand_like(param) < mutation_rate
                # Add random noise to selected weights
                noise = torch.randn_like(param) * mutation_strength
                param[mask] += noise[mask]
        
        self.metadata['mutations_applied'] += 1
        self.metadata['evaluated'] = False  # Reset fitness evaluation
    
    def crossover(self, other, crossover_rate=0.5):
        """
        Perform crossover with another individual to create offspring.
        TODO: Implement more sophisticated crossover strategies.
        
        Args:
            other: Another Individual to crossover with
            crossover_rate: Probability of taking weights from 'other' (default: 0.5)
            
        Returns:
            New Individual (offspring)
        """
        # Create new model with same architecture
        # offspring_model = CreateModel(
        #     self.architecture['input_features'] or 4096,
        #     self.architecture['h1'] or 512,
        #     self.architecture['h2'] or 256,
        #     self.architecture['h3'] or 128,
        #     self.architecture['output_features'] or 3,
        #     verbose=self.model.verbose
        # )

        # Temporarily use SimpleNet as the offspring model
        offspring_model = SimpleNet()
        
        # Perform uniform crossover on weights
        with torch.no_grad():
            for param_self, param_other, param_offspring in zip(
                self.model.parameters(),
                other.model.parameters(),
                offspring_model.parameters()
            ):
                # Random mask: True = take from other, False = take from self
                mask = torch.rand_like(param_self) < crossover_rate
                param_offspring.data = param_self.data.clone()
                param_offspring.data[mask] = param_other.data[mask]
        
        # Create offspring Individual
        offspring = Individual(model=offspring_model, verbose=self.model.verbose)
        offspring.generation = max(self.generation, other.generation) + 1
        offspring.parent_ids = [
            self.individual_id if self.individual_id else None,
            other.individual_id if other.individual_id else None
        ]
        offspring.metadata['crossover_count'] = 1
        
        # Update parent crossover counts
        self.metadata['crossover_count'] += 1
        other.metadata['crossover_count'] += 1
        
        return offspring
    
    def save(self, file_path):
        """
        Save the individual's model.
        
        Args:
            file_path: Full path to save the model (including filename, with or without .pth extension)
            
        Returns:
            Full path where the model was saved
        """
        # Extract directory and filename
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # Create directory if it doesn't exist
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Add .pth extension if not present
        if not filename.endswith('.pth'):
            file_path = file_path + '.pth'
        
        # Save the model's state_dict
        torch.save(self.model.state_dict(), file_path)
        
        return file_path
    
    def load(self, model_path):
        """
        Load model weights from a saved file.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model.load_state_dict(torch.load(model_path))
        self.metadata['evaluated'] = False
    
    # Delegate model methods for convenience
    def test(self, X):
        """Test the model (delegates to model.test)."""
        return self.model.test(X)
    
    def forward(self, x):
        """Forward pass (delegates to model.forward)."""
        return self.model(x)
    
    def __repr__(self):
        """String representation of the Individual."""
        fitness_str = f"fitness={self.fitness:.6f}" if self.fitness is not None else "fitness=None"
        id_str = f"id={self.individual_id}" if self.individual_id else "id=None"
        return f"Individual({id_str}, {fitness_str}, gen={self.generation})"
    
    def __lt__(self, other):
        """Comparison for sorting (higher fitness is better)."""
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        # Higher fitness is better, so reverse the comparison
        return self.fitness > other.fitness
    
    def __eq__(self, other):
        """Equality comparison."""
        return self.individual_id == other.individual_id if isinstance(other, Individual) else False




# # Load the dummy data
# X, y_command, y_continuous = load_data('dummy_data.json')

# # Create the model
# test_model = CreateModel(input_features=4096, h1=512, h2=256, h3=128, output_features=3)

# # Create trainer and train
# trainer = ModelTrainer(test_model, learning_rate=0.001)
# trainer.train(X, y_command, epochs=100)



