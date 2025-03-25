# RL Sensitivity Analysis

## Overview
This repository contains the implementation, analysis, and report for our Reinforcement Learning (RL) Sensitivity Analysis project. The project's focus is solving the **MountainCarContinuous-v0** environment from the Gymnasium and performing sensitivity analyses on different hyperparameters. We implemented several RL techniques and enhancements to address the challenges of continuous control.

## Assignment Details
- **Part 1: Environment Analysis**  
  A detailed description of the MountainCarContinuous-v0 environment, including its continuous action space, observation structure (position and velocity), reward signals, and dynamics.
  
- **Part 2: DQN Implementation**  
  Implementation of the Deep Q-Network (DQN) algorithm using PyTorch. Custom environment wrappers were created to adapt the continuous action space for DQN by discretizing actions and normalizing observations.

- **Part 3: Sensitivity Analysis**  
  Investigation of key hyperparameters (learning rate, exploration decay, batch size, discretization bins, network architecture, etc.) and their effect on training stability and performance.

- **Part 4: Advanced DQN Improvements**  
  Enhance basic DQN with techniques such as Prioritized Experience Replay (PER), Double DQN, Dueling Architecture, and Noisy Networks to improve learning stability and efficiency.

- **Part 5: Neural Network Structure Exploration**  
  Experimentation with different neural network architectures (LSTM, GRU, and Transformer) integrated within the DQN framework to assess their impact on learning in a sequential decision-making environment.

- **Part 6: PPO and A2C Comparison**  
  Implementation and evaluation of policy gradient methods like Proximal Policy Optimization (PPO) and Synchronous Advantage Actor-Critic (A2C) for a comparative study against the value-based methods.

## Environment: MountainCarContinuous-v0
- **Action Space:**  
  Continuous scalar value in the range [-1.0, 1.0] representing the force applied to the car.
  
- **Observation Space:**  
  A 2-dimensional vector representing:
  - **Position:** Car's horizontal location.
  - **Velocity:** Rate of change of position.
  
- **Reward Signal:**  
  A negative penalty for large actions combined with a large positive bonus when the goal is reached.

## How to Run
1. **Installation:**  
   Ensure you have Python 3.7+ installed.

2. **Running the Notebooks:**  
   Since the project uses Jupyter Notebooks for running experiments and analysis, follow these steps:

  
- Open a terminal or command prompt.

  
- Navigate to the project directory.

  
- Launch Jupyter Notebook or Jupyter Lab.

- Open and run the following notebooks:

    - Assignment RL 2025 (DQN Analysis).ipynb

    - PPO_A2C_Analysis.ipynb

- Follow the instructions within the notebooks to execute the cells and view the analysis.

3. **Results:**  
   Training logs, plots, and performance metrics will be saved in the results/ folder and inside the Jupyter notebooks. Detailed analysis can be found in the report.

## Acknowledgements
### Authors:

**Kostoulas Evangelos**

**Kotsopoulos Dimitris**

**Triantafillou Thanasis**

## License

This project is licensed under the MIT License.