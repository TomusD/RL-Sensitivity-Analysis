import os
import time
import torch
import pygame
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=0):
    """
    Set seed for reproducibility across multiple libraries and environments.

    Args:
        seed (int): The seed value to set. Default is 0.
    """
    # Seed NumPy
    np.random.seed(seed)
    np.random.default_rng(seed)
    
    # Seed Python's built-in hash function
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed PyTorch on CPU
    torch.manual_seed(seed)
    
    # Seed PyTorch on GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic results

class ReplayMemory:
    """
    Replay memory using pre-allocated NumPy arrays for efficiency.
    """

    def __init__(self, capacity, state_dim, action_dim, device):
        """
        Initialize the ReplayMemory with fixed-size arrays.

        Args:
            capacity (int): Maximum number of transitions to store.
            state_dim (int or tuple): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            device (torch.device): Device to which the sampled tensors should be moved.
        """
        self.capacity = capacity
        self.device = device
        self.current_idx = 0
        self.is_full = False

        # Pre-allocate memory for transitions
        self.states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.next_states = np.zeros((capacity, *state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminateds = np.zeros((capacity,), dtype=bool) 

    def store(self, state, action, next_state, reward, terminated):
        """
        Store a transition in the replay memory.

        Args:
            state (np.ndarray): Current state.
            action (int or float): Action taken.
            next_state (np.ndarray): Next state after the action.
            reward (float): Reward received.
            terminated (bool): Whether the episode terminated (natural end).
        """
        self.states[self.current_idx] = state
        self.actions[self.current_idx] = action
        self.next_states[self.current_idx] = next_state
        self.rewards[self.current_idx] = reward
        self.terminateds[self.current_idx] = terminated  

        # Update the index and handle buffer wrapping
        self.current_idx += 1
        if self.current_idx == self.capacity:
            self.is_full = True
            self.current_idx = 0

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple[torch.Tensor]: Sampled transitions (states, actions, next_states, rewards, terminateds).
        """
        upper_bound = self.capacity if self.is_full else self.current_idx
        indices = np.random.choice(upper_bound, size=batch_size, replace=False)

        # Convert sampled data to PyTorch tensors and move to the device
        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        terminateds = torch.as_tensor(self.terminateds[indices], device=self.device)

        return states, actions, next_states, rewards, terminateds

    def __len__(self):
        """
        Return the current number of stored transitions.

        Returns:
            int: Number of stored transitions.
        """
        return self.capacity if self.is_full else self.current_idx

class DQN_Network(nn.Module):
    """
    The Deep Q-Network (DQN) model for reinforcement learning.
    
    This network approximates the Q-value function, which predicts the expected 
    cumulative reward for each possible action in a given state. The architecture 
    consists of fully connected (FC) layers with ReLU activation functions.
    """

    def __init__(self, num_actions, input_dim, hidden_dim=64):
        """
        Initialize the DQN network.

        Args:
            num_actions (int): The number of possible actions in the environment. 
                               This determines the size of the output layer.
            input_dim (int): The dimensionality of the input state space, i.e., the 
                             number of features describing the state.
            hidden_dim (int, optional): The number of units in the hidden layer. 
                                        Default is 64.
        """
        super(DQN_Network, self).__init__()
        
        # Define the fully connected layers of the network
        self.FC = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input to hidden layer
            nn.ReLU(inplace=True),            # Activation function
            nn.Linear(hidden_dim, num_actions)  # Hidden to output layer
        )
        
        # Apply Xavier Initialization to all Linear layers
        self.FC.apply(self.init_weights)

    def init_weights(layer):
        """
        Initialize weights for Linear layers using Xavier Uniform initialization.
        
        Xavier initialization ensures that the variance of the activations remains 
        consistent across layers, preventing vanishing or exploding gradients. 
        This is particularly effective for activation functions like ReLU.
        
        Args:
            layer (nn.Module): A layer in the neural network. If it is a Linear 
                               layer, its weights and biases are initialized.
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # Initialize weights
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # Initialize biases to zero

    def forward(self, x):
        """
        Perform the forward pass to compute Q-values for each action.

        The forward pass processes the input state through the fully connected 
        layers, applying the ReLU activation in the hidden layer, and outputs 
        the Q-values for all possible actions.

        Args:
            x (torch.Tensor): A tensor representing the state of the environment. 
                              Shape: (batch_size, input_dim).

        Returns:
            torch.Tensor: A tensor containing Q-values for each action. 
                          Shape: (batch_size, num_actions).
        """
        return self.FC(x)
    
class DQN_Agent:
    """
    DQN Agent for reinforcement learning.
    
    This class implements the core components of the Deep Q-Network (DQN) algorithm, 
    including epsilon-greedy action selection, learning updates, target network synchronization, 
    and experience replay.
    """

    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay, clip_grad_norm, 
                 learning_rate, discount, memory_capacity, seed, network_class=DQN_Network):
        """
        Initialize the DQN agent.

        Args:
            env (gym.Env): The environment in which the agent operates.
            epsilon_max (float): Initial value of epsilon for the epsilon-greedy policy.
            epsilon_min (float): Minimum value of epsilon for the epsilon-greedy policy.
            epsilon_decay (float): Decay rate for epsilon.
            clip_grad_norm (float): Maximum value for gradient clipping to prevent exploding gradients.
            learning_rate (float): Learning rate for the optimizer.
            discount (float): Discount factor for future rewards (gamma).
            memory_capacity (int): Capacity of the replay memory buffer.
        """
        self.loss_history = []  # Tracks the loss per episode
        self.running_loss = 0   # Accumulates loss during an episode
        self.learned_counts = 0  # Tracks the number of updates per episode

        # Reinforcement learning hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)  # Set the seed for reproducible action sampling
        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity, self.observation_space.shape, 1, device)

        # Initialize the main and target networks
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.n
        self.main_network = network_class(num_actions=output_dim, input_dim=input_dim).to(device)
        self.target_network = network_class(num_actions=output_dim, input_dim=input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        # Loss function and optimizer
        self.clip_grad_norm = clip_grad_norm
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        """
        Select an action based on the epsilon-greedy policy.

        Args:
            state (torch.Tensor): Input tensor representing the current state.

        Returns:
            int: The selected action.
        """
        # Exploration: Select a random action
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        # Exploitation: Select the action with the highest Q-value
        if not torch.is_tensor(state):  # Ensure the state is a tensor
            state = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()
        return action

    def learn(self, batch_size, done):
        """
        Perform a learning step to update the main network's parameters.

        Args:
            batch_size (int): Number of samples to draw from the replay memory.
            done (bool): Whether the current episode is finished. Used for logging purposes.
        """
        # Sample a batch of transitions from the replay memory
        states, actions, next_states, rewards, terminateds = self.replay_memory.sample(batch_size)

        rewards = rewards.unsqueeze(1)
        terminateds = terminateds.unsqueeze(1)

        # Compute Q-values for the selected actions
        predicted_q = self.main_network(states).gather(dim=1, index=actions)

        # Compute the target Q-values using the target network
        with torch.no_grad():
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            next_target_q_value[terminateds] = 0  # No future rewards for terminal states

        # Bellman equation for target Q-values
        y_js = rewards + (self.discount * next_target_q_value)

        # Compute the loss
        loss = self.criterion(predicted_q, y_js)

        # Update running loss and log loss at the end of the episode
        self.running_loss += loss.item()
        self.learned_counts += 1
        if done:
            self.loss_history.append(self.running_loss / self.learned_counts)
            self.running_loss = 0
            self.learned_counts = 0

        # Optimize the main network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def update_target_network(self):
        """
        Update the target network by copying parameters from the main network.

        This is a "hard" update, where the target network parameters are directly 
        replaced with the main network's parameters.
        """
        self.target_network.load_state_dict(self.main_network.state_dict())

    def update_epsilon(self):
        """
        Update epsilon for the epsilon-greedy exploration strategy.

        The value of epsilon decreases over time, reducing exploration and increasing exploitation.
        """
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save_checkpoint(self, path):
        """
        Save the main network's parameters and training metadata to a checkpoint file.

        Args:
            path (str): Path to the file where the checkpoint will be saved.
        """
        checkpoint = {
            "main_network_state_dict": self.main_network.state_dict(),  # Model weights
            "optimizer_state_dict": self.optimizer.state_dict(),        # Optimizer state
            "loss_history": self.loss_history,                           # Loss history for logging
            "learning_rate": self.optimizer.defaults['lr'],             # Current learning rate
        }
        torch.save(checkpoint, path)

class step_wrapper(gym.Wrapper):
    """
    A wrapper for customizing the behavior of an environment's step and reset methods.

    This wrapper applies modifications to the state and reward functions returned by 
    the `step` and `reset` methods using the `observation_wrapper` and `reward_wrapper`.

    Args:
        env (gym.Env): The environment to be wrapped.
    """
    def __init__(self, env):
        """
        Initialize the step wrapper with the environment to be wrapped.

        Args:
            env (gym.Env): The environment to be wrapped.
        """
        super().__init__(env)
        self.observation_wrapper = observation_wrapper(env)
        self.reward_wrapper = reward_wrapper(env)

    def step(self, action):
        """
        Execute a step in the environment with a modified state and reward.

        Args:
            action (int): The action to be taken.

        Returns:
            Tuple: Modified state, modified reward, done flag, truncation flag, and info.
        """
        state, reward, done, truncation, info = self.env.step(action)
        modified_state = self.observation_wrapper.observation(state)
        modified_reward = self.reward_wrapper.reward(modified_state, action)
        return modified_state, modified_reward, done, truncation, info

    def reset(self, seed=None):
        """
        Reset the environment and return the modified initial state.

        Args:
            seed (int, optional): Random seed for the environment reset.

        Returns:
            Tuple: Modified initial state and additional info.
        """
        state, info = self.env.reset(seed=seed)
        modified_state = self.observation_wrapper.observation(state)
        return modified_state, info


class observation_wrapper(gym.ObservationWrapper):
    """
    A wrapper for modifying observations in the environment.

    This wrapper normalizes observations using min-max scaling for improved compatibility
    with neural network inputs.

    Args:
        env (gym.Env): The environment to wrap.
    """
    def __init__(self, env):
        """
        Initialize the observation wrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super().__init__(env)
        self.min_value = env.observation_space.low
        self.max_value = env.observation_space.high

    def observation(self, state):
        """
        Normalize the observation using min-max scaling.

        Args:
            state (numpy.ndarray): The original observation from the environment.

        Returns:
            numpy.ndarray: The normalized observation.
        """
        return (state - self.min_value) / (self.max_value - self.min_value)


class reward_wrapper(gym.RewardWrapper):
    """
    A wrapper for customizing rewards in the environment.

    This wrapper penalizes large actions and provides a bonus for achieving the goal.

    Args:
        env (gym.Env): The environment to wrap.
    """
    def __init__(self, env):
        """
        Initialize the reward wrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super().__init__(env)

    def reward(self, state, action):
        """
        Compute a modified reward based on the current state and action.

        - Applies a penalty proportional to the square of the action magnitude.
        - Rewards the agent with a bonus for reaching the goal.

        Args:
            state (numpy.ndarray): The state of the environment.
            action (int): The action taken.

        Returns:
            float: The modified reward.
        """
        position, velocity = state
        continuous_action = self.env.discrete_actions[action]

        if position >= 0.9167:  # Goal position
            return 100.0 - 0.1 * (continuous_action ** 2)  # Goal reward with penalty

        return -0.1 * (continuous_action ** 2)  # Regular penalty


class DiscretizedActionWrapper(gym.ActionWrapper):
    """
    A wrapper to discretize the continuous action space of the environment.

    Converts the continuous action space into a finite number of bins for environments
    like `MountainCarContinuous-v0`.

    Args:
        env (gym.Env): The environment to wrap.
        num_bins (int, optional): Number of discrete bins. Default is 20.
    """
    def __init__(self, env, num_bins=20):
        """
        Initialize the action wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            num_bins (int, optional): Number of discrete bins. Default is 20.
        """
        super().__init__(env)
        self.num_bins = num_bins
        self.action_space = gym.spaces.Discrete(num_bins)
        self.discrete_actions = np.linspace(env.action_space.low[0], env.action_space.high[0], num_bins)

    def action(self, action):
        """
        Map a discrete action to its corresponding continuous value.

        Args:
            action (int): The discrete action index.

        Returns:
            list: The corresponding continuous action value.
        """
        return [self.discrete_actions[action]]

class DQN_Train:
    """
    A class to train and test a Deep Q-Network (DQN) agent on a given environment.

    This class manages the training and testing process of a reinforcement learning agent, including:
    - Managing training loop and epsilon-greedy strategy.
    - Saving and loading checkpoints.
    - Plotting training performance metrics such as rewards and network loss.

    Args:
        agent (DQN_Agent): The agent to be trained or tested.
        env (gym.Env): The environment in which the agent operates.
        hyperparams (dict): A dictionary containing the hyperparameters for training and testing.
    """

    def __init__(self, agent, env, hyperparams):
        """
        Initialize the DQN training class with the agent, environment, and hyperparameters.

        Args:
            agent (DQN_Agent): The DQN agent to be trained or tested.
            env (gym.Env): The environment in which the agent operates.
            hyperparams (dict): A dictionary containing the training parameters.
        """
        self.agent = agent
        self.env = env
        self.RL_load_path = hyperparams["load_path"]
        self.save_path = hyperparams["save_path"]
        self.save_interval = hyperparams["save_interval"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.seed = hyperparams["seed"]
        self.timings = []  # List to store timing information for intervals

    def run(self, mode='train', max_episodes=10):
        """
        Run the training or testing loop based on the specified mode.

        Args:
            mode (str): The mode of operation. 'train' for training, 'test' for testing. Default is 'train'.
            max_episodes (int): The number of episodes to run. Used only in 'test' mode. Default is 10.
        """
        assert mode in ['train', 'test'], "Mode must be 'train' or 'test'."

        total_steps = 0
        self.reward_history = []  # To keep track of rewards in each episode
        interval_start_time = time.time()  # Start timer for interval timing

        # Load checkpoint and prepare the model for testing
        if mode == 'test':
            max_episodes = max_episodes  # Override the max episodes for testing
            checkpoint = torch.load(self.RL_load_path, weights_only=True)
            self.agent.main_network.load_state_dict(checkpoint["main_network_state_dict"])
            self.agent.main_network.eval()

            # Print checkpoint details
            print("Checkpoint loaded:")
            print(f"Learning rate: {checkpoint['learning_rate']}")
            print(f"Loss history (last 5 entries): {checkpoint['loss_history'][-5:]}")

            # Get optimizer state information
            optimizer_state = checkpoint['optimizer_state_dict']
            print(f"Optimizer state - Learning rate: {optimizer_state['param_groups'][0]['lr']}")
            print(f"Optimizer state - Beta1: {optimizer_state['param_groups'][0]['betas'][0]}")
            print(f"Optimizer state - Beta2: {optimizer_state['param_groups'][0]['betas'][1]}")
            print("-----------------------------------------------")

        # Main loop for training or testing
        for episode in range(1, (max_episodes if mode == 'test' else self.max_episodes) + 1):
            state, _ = self.env.reset(seed=None if mode == 'test' else self.seed)
            done = False
            truncation = False
            step_size = 0
            episode_reward = 0

            while not done and not truncation:
                action = self.agent.select_action(state)
                next_state, reward, done, truncation, _ = self.env.step(action)

                # Training phase: Store experience and perform learning updates
                if mode == 'train':
                    self.agent.replay_memory.store(state, action, next_state, reward, done)
                    if len(self.agent.replay_memory) > self.batch_size:
                        self.agent.learn(self.batch_size, (done or truncation))
                        if total_steps % self.update_frequency == 0:
                            self.agent.update_target_network()

                state = next_state
                episode_reward += reward
                step_size += 1

            # Handle training and logging
            if mode == 'train':
                self.reward_history.append(episode_reward)
                total_steps += step_size
                self.agent.update_epsilon()

                if episode % self.save_interval == 0:
                    self.agent.save_checkpoint(self.save_path + '_' + f'{episode}' + '.pth')
                    elapsed_time = time.time() - interval_start_time
                    self.timings.append(elapsed_time)
                    interval_start_time = time.time()  # Reset the interval timer
                    print(f'\nModel saved. Time for last {self.save_interval} episodes: {elapsed_time:.2f} seconds.\n')
                    if episode != self.max_episodes:
                        self.plot_training(episode)

                print(f"Episode: {episode}, Total Steps: {total_steps}, Ep Step: {step_size}, "
                      f"Raw Reward: {episode_reward:.2f}, Epsilon: {self.agent.epsilon_max:.2f}")

            else:  # Test mode
                print(f"Test Episode: {episode}, Steps: {step_size}, Reward: {episode_reward:.2f}")

        # Plot training results after the loop
        if mode == 'train':
            self.plot_training(episode)


    def plot_training(self, episode):
        """
        Plots the training statistics for the current episode, including:
        - Raw rewards and their Simple Moving Average (SMA) with a window of 50 episodes.
        - Network loss over episodes.
    
        This function saves the combined plot as an image, displaying the raw rewards 
        and the network loss for a specific episode, as well as calculating and visualizing
        the SMA for rewards.

        Args:
            episode (int): The current episode number for which the training statistics are plotted.
        """
        
        # Create the folder if it doesn't exist
        plot_dir = os.path.dirname(self.save_path)
        os.makedirs(plot_dir, exist_ok=True)

        # Calculate the Simple Moving Average (SMA) with a window size of 50
        sma = np.convolve(self.reward_history, np.ones(50) / 50, mode='same')

        # Determine y-axis max for rewards
        max_reward = max(np.max(self.reward_history), np.max(sma))
        # Determine y-axis min for rewards
        min_reward = min(np.min(self.reward_history), np.min(sma))

        # Create a figure with two subplots (1 row, 2 columns)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot rewards
        ax1.set_title(f"Obtained Rewards - Episode {episode}")
        ax1.plot(self.reward_history, label='Raw Reward', color='#1f77b4', alpha=1)
        ax1.plot(sma, label='Simple Moving Average 50', color='#ff7f0e')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Rewards")
        ax1.set_ylim(min_reward - 10, max_reward + 10)  # Add some padding to the top
        ax1.legend()
        ax1.grid(True)

        # Plot network loss
        max_loss = max(self.agent.loss_history) if self.agent.loss_history else 0
        ax2.set_title(f"Network Loss - Episode {episode}")
        ax2.plot(self.agent.loss_history, label='Loss', color='#2ca02c', alpha=1)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.set_ylim(0, max_loss + 10)  # Add some padding to the top
        ax2.legend()
        ax2.grid(True)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Add timing text below the plots
        timing_text = f"Last Interval ({(episode - self.save_interval + 1)}-{episode}): {self.timings[-1]:.2f} seconds"
        plt.figtext(0.5, -0.05, timing_text, wrap=True, horizontalalignment='center', fontsize=8)

        # Save the combined plot with the episode number
        plot_path = os.path.join(plot_dir, f'combined_plot_episode_{episode}.png')
        plt.savefig(plot_path, format='png', dpi=600, bbox_inches='tight')

        # Display the plot
        plt.show()
        plt.clf()
        plt.close()


if __name__ == '__main__':
    # Set seed for reproducibility across experiments
    seed = 0
    set_seed(seed)

    # Set up the path for saving model checkpoints
    save_dir = "saved models"
    # Ensure the "saved models" directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Parameters for training or testing
    train_mode = True  # Flag to toggle between training and testing modes
    render = not train_mode  # If not training, render the environment
    hyperparams = {
        "load_path": None,  # Path to load the pre-trained model, None during training
        "save_path": os.path.join(save_dir, "model_checkpoint"),  # Path to save model checkpoints
        "save_interval": 100,  # Interval at which the model will be saved (in episodes)
        "clip_grad_norm": 5,  # Maximum gradient norm for clipping during training
        "learning_rate": 0.01,  # Learning rate for training the agent
        "discount_factor": 0.99,  # Discount factor (gamma) for future rewards
        "batch_size": 128,  # Batch size for training updates
        "update_frequency": 50,  # Number of steps before updating the target network
        "max_episodes": 2000,  # Maximum number of episodes for training
        "max_steps": 1000,  # Maximum number of steps per episode
        "epsilon_max": 0.999 if train_mode else 0,  # Exploration rate (epsilon) for epsilon-greedy policy
        "epsilon_min": 0.01,  # Minimum exploration rate
        "epsilon_decay": 0.998,  # Decay rate for epsilon
        "memory_capacity": 250_000 if train_mode else 0,  # Capacity for replay memory (only used in training)
        "render_fps": 60,  # Frames per second for rendering the environment
        "seed": seed  # Random seed for reproducibility
    }

    # Initialize the environment (MountainCarContinuous)
    env = gym.make("MountainCarContinuous-v0", max_episode_steps=hyperparams["max_steps"],
                   render_mode="human" if render else None)  # Set render mode based on the train_mode
    env.metadata["render_fps"] = hyperparams["render_fps"]  # Set render FPS for the environment

    # Apply wrappers to the environment
    env = DiscretizedActionWrapper(env, num_bins=20)  # Discretize the continuous action space
    env = step_wrapper(env)  # Apply step wrapper for observation and reward modifications

    # Initialize the agent with the provided hyperparameters
    agent = DQN_Agent(env=env,
                      epsilon_max=hyperparams["epsilon_max"],
                      epsilon_min=hyperparams["epsilon_min"],
                      epsilon_decay=hyperparams["epsilon_decay"],
                      clip_grad_norm=hyperparams["clip_grad_norm"],
                      learning_rate=hyperparams["learning_rate"],
                      discount=hyperparams["discount_factor"],
                      memory_capacity=hyperparams["memory_capacity"],
                      seed=seed)

    # Initialize the training or testing procedure
    dqn_trainer = DQN_Train(agent=agent, env=env, hyperparams=hyperparams)

    # Run the trainer in the desired mode (train or test)
    if train_mode:
        # Create the "saved models" folder if it does not exist
        save_dir = "saved models"
        os.makedirs(save_dir, exist_ok=True)
        dqn_trainer.run(mode='train')  # Train the model
    else:
        # Loop through checkpoints for testing
        first_model = 400  # Starting checkpoint for testing
        last_model = 500   # Ending checkpoint for testing
        for checkpoint in range(first_model, last_model, hyperparams["save_interval"]):  # Iterate through checkpoints
            checkpoint_path = os.path.join(save_dir, f"model_checkpoint_{checkpoint}.pth")  # Path to the current checkpoint
            print(f"\nTesting with checkpoint: {checkpoint_path}")

            # Update the load path in hyperparameters for the current checkpoint
            hyperparams["load_path"] = checkpoint_path

            # Reinitialize the DQN trainer with updated hyperparameters
            dqn_trainer = DQN_Train(agent=agent, env=env, hyperparams=hyperparams)

            # Run the test mode for the current checkpoint
            dqn_trainer.run(mode='test', max_episodes=2)  # Test with a limited number of episodes (2)

        pygame.quit()  # Close the pygame window after testing is complete