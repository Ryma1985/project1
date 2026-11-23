import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import sys
from collections import deque
import shap
from sklearn.model_selection import train_test_split


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQNAgent to handle action selection and learning
class DDQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.00001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update_freq=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = deque(maxlen=10000)

        # Initialize Q-network and target network
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)

        # Synchronize target network with the Q-network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.target_update_freq = target_update_freq  # Frequency for updating the target network
        self.update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state)
            return torch.argmax(action_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None  # Not enough data in memory

        batch = random.sample(self.memory, batch_size)
        total_loss = 0

        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            # DDQN: Use the main model to select the best action
            with torch.no_grad():
                best_action = torch.argmax(self.model(next_state)).item()
                target_q_value = reward + (1 - done) * self.gamma * self.target_model(next_state)[best_action]

            # Compute current Q-value
            current_q_value = self.model(state)[action]

            # Compute loss
            loss = self.loss_fn(current_q_value, target_q_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return total_loss / batch_size


    def explain_action(self, state, action_index=0):
        # Ensure state is reshaped to a 2D array for SHAP compatibility
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        state_np = state_tensor.detach().numpy()

        # Define a wrapper function for the model's output for the chosen action
        def model_fn(x):
            x_tensor = torch.FloatTensor(x)
            with torch.no_grad():
                return self.model(x_tensor).detach().numpy()[:, action_index]

        # Use KernelExplainer for SHAP explanations
        explainer = shap.KernelExplainer(model_fn, np.zeros((1, len(state_np[0]))))
        shap_values = explainer.shap_values(state_np)

        # Return shap_values[0] only if it's non-empty
        return shap_values[0] if shap_values[0].size > 0 else None
    
    
class Simulation:
    def __init__(self, taxi_data_path, vehicles_data_path, grid_size=(20, 20), episode_duration=40):
        # Load taxi data
        self.taxi_data = pd.read_csv(taxi_data_path, delimiter=',')
        self.taxi_data['tpep_pickup_datetime'] = pd.to_datetime(self.taxi_data['tpep_pickup_datetime'])
        self.taxi_data['tpep_dropoff_datetime'] = pd.to_datetime(self.taxi_data['tpep_dropoff_datetime'])

        # Split data into training, validation, and testing sets
        train_val_data, self.test_data = train_test_split(self.taxi_data, test_size=0.2, random_state=42)
        self.train_data, self.val_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

        # Load vehicle data
        self.vehicles_data = pd.read_csv(vehicles_data_path)
        self.location_to_cluster = {vehicle['Nodelocation']: vehicle['Cluster_ID'] for _, vehicle in self.vehicles_data.iterrows()}
        
        # Initialize grid and other variables
        self.grid_size = grid_size
        self.episode_duration = episode_duration
        self.timestep_duration = 10
        self.timesteps_per_episode = self.episode_duration // self.timestep_duration
        self.idle_vehicle_grid = np.zeros(grid_size)
        self.vehicle_status = ['idle'] * len(self.vehicles_data)
        self.service_timers = [0] * len(self.vehicles_data)

        # Initialize idle vehicles in their clusters
        for position in self.vehicles_data['Cluster_ID']:
            row, col = eval(position)
            self.idle_vehicle_grid[row, col] += 1

        self.vehicle_assignments_df = pd.DataFrame(columns=['Vehicle_ID', 'Order_ID', 'Assigned_Cluster_ID', 'Dropoff_Cluster_ID'])

        self.start_time = self.taxi_data['tpep_pickup_datetime'].min()  # Earliest pickup time in dataset
        self.current_time = self.start_time  

        # Initialize DQN agent
        self.agent = DDQNAgent(state_dim=18, action_dim=9)

      # Separate storage for metrics
        self.training_rewards = []  # Cumulative rewards for training
        self.validation_rewards = []  # Cumulative rewards for validation
        self.testing_rewards = []  # Cumulative rewards for testing
        self.training_rejection_rates = []  # Rejection rates for training
        self.validation_rejection_rates = []  # Rejection rates for validation
        self.testing_rejection_rates = []  # Rejection rates for testing

        # Initialize rejection rate metrics
        self.rejection_rate_timestep = []  # Tracks rejection rates at each timestep
        self.episode_rejection_rate = []  # Tracks rejection rates at each episode
       # self.cumulative_rewards = []
        # Loss tracking
        self.step_losses = []  # Training loss per step
        self.validation_losses = []  # Validation loss per episode
       
    def get_neighbors(self, row, col):
        neighbors = []
        for r in range(max(0, row - 1), min(self.grid_size[0], row + 2)):
            for c in range(max(0, col - 1), min(self.grid_size[1], col + 2)):
                if (r, c) != (row, col):
                    neighbors.append((r, c))
        return neighbors

   
  
    def get_state(self, row, col, pending_order_grid):
    
        state = [pending_order_grid[row, col], self.idle_vehicle_grid[row, col]]
        neighbors = self.get_neighbors(row, col)

        for n_row, n_col in neighbors:
            
            state.extend([pending_order_grid[n_row, n_col], self.idle_vehicle_grid[n_row, n_col]])

        # Pad with zeros if fewer than 8 neighbors
        while len(state) < 18:
            state.extend([0, 0])

        return np.array(state)

   
    def dispatch_orders(self, pending_order_grid, timestep_orders):
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # Get the initial state for the current cluster
                initial_state = self.get_state(row, col, pending_order_grid)
                print(f"Initial State of cluster {row},{col}: {initial_state}")
                self.agent.explain_action(initial_state)
                # Process orders in the current cluster
                while pending_order_grid[row, col] > 0:
                    # Get the valid actions for this cluster
                    neighbors = self.get_neighbors(row, col)
                    valid_actions = [0] + list(range(1, len(neighbors) + 1))  # 0 = stay, 1 to N = neighbors
                    print(f"Valid actions for cluster ({row},{col}): {valid_actions}")

                    assigned = False  # Flag to track if the order gets assigned
                    action = self.agent.act(initial_state) % len(valid_actions)  # Agent selects action
                    reward = self.calculate_reward(action, row, col, pending_order_grid)  # Calculate reward

                    if action == 0:  # Stay in the current cluster
                        if self.idle_vehicle_grid[row, col] > 0:
                            # Assign vehicle within the same cluster
                            self.idle_vehicle_grid[row, col] -= 1
                            pending_order_grid[row, col] -= 1
                            assigned_order = timestep_orders.iloc[0]
                            timestep_orders = timestep_orders.iloc[1:]
                            trip_duration = (assigned_order['tpep_dropoff_datetime'] - assigned_order['tpep_pickup_datetime']).total_seconds() / 60.0
                            # Assign the first idle vehicle in the current cluster
                            idle_vehicles_idx = np.where((np.array(self.vehicle_status) == 'idle') &
                                                     (self.vehicles_data['Cluster_ID'] == f"({row},{col})"))[0]
                            if len(idle_vehicles_idx) > 0:
                                vehicle_idx = idle_vehicles_idx[0]
                                self.vehicle_status[vehicle_idx] = 'busy'
                                self.service_timers[vehicle_idx] = trip_duration
                            else:
                                vehicle_idx = None  # Default value if no idle vehicles are found

                            if vehicle_idx is not None:
                                assigned = True
                                print(f"Action: Stay in cluster ({row},{col}), reward: {reward}")
                            else:
                                print(f"No idle vehicles available in cluster ({row},{col}).")
                    else:
                        # Move to a neighboring cluster
                        if action - 1 < len(neighbors):
                            n_row, n_col = neighbors[action - 1]
                            if self.idle_vehicle_grid[n_row, n_col] > 0:
                                # Assign the first available idle vehicle from a neighboring cluster
                                neighbor_idle_vehicles_idx = np.where((np.array(self.vehicle_status) == 'idle') &
                                                                  (self.vehicles_data['Cluster_ID'] == f"({n_row},{n_col})"))[0]
                                if len(neighbor_idle_vehicles_idx) > 0:
                                    vehicle_idx = neighbor_idle_vehicles_idx[0]
                                    self.idle_vehicle_grid[n_row, n_col] -= 1
                                    pending_order_grid[row, col] -= 1
                                    assigned_order = timestep_orders.iloc[0]
                                    timestep_orders = timestep_orders.iloc[1:]
                                    trip_duration = (assigned_order['tpep_dropoff_datetime'] - assigned_order['tpep_pickup_datetime']).total_seconds() / 60.0
                                    self.vehicle_status[vehicle_idx] = 'busy'
                                    self.service_timers[vehicle_idx] = trip_duration
                                else:
                                    vehicle_idx = None  # Default value if no idle vehicles are found

                                if vehicle_idx is not None:
                                    assigned = True
                                    print(f"Action: Move to neighbor ({n_row},{n_col}), reward: {reward}")
                                else:
                                    print(f"No idle vehicles available in neighboring cluster ({n_row},{n_col}).")

                    if not assigned:
                        print(f"No idle vehicles available for cluster ({row},{col}) or neighbors.")
                        break

                    # Update state after the action
                    next_state = self.get_state(row, col, pending_order_grid)
                    self.agent.remember(initial_state, action, reward, next_state, done=False)
                    initial_state = next_state

                    print(f"Action: {action}, Reward: {reward}, Next State: {next_state}")

                    if assigned:
                        # Record the assignment
                        self.vehicle_assignments_df = self.vehicle_assignments_df.append({
                            'Vehicle_ID': vehicle_idx,
                            'Order_ID': assigned_order.name,
                            'Assigned_Cluster_ID': (row, col),
                            'Dropoff_Cluster_ID': 'N/A'
                        }, ignore_index=True)
                        print(f"Vehicle {vehicle_idx} assigned to order in cluster {row},{col} and will be busy for {trip_duration} minutes.")
                    else:
                        # No idle vehicles available
                        print(f"No idle vehicles available for order in cluster ({row},{col}) or its neighbors.")
                        break

    def update_vehicle_status(self, timestep_duration):
        for i in range(len(self.vehicles_data)):
            if self.vehicle_status[i] == 'busy':
                remaining_service_time = self.service_timers[i]
                self.service_timers[i] -= min(timestep_duration, remaining_service_time)

                # Check if the vehicle has completed its trip
                if self.service_timers[i] <= 0:  # Trip complete
                    last_order = self.vehicle_assignments_df[self.vehicle_assignments_df['Vehicle_ID'] == i].iloc[-1]
                    assigned_order_id = last_order['Order_ID']
                
                    # Retrieve the assigned order details
                    assigned_order = self.taxi_data.loc[assigned_order_id]
                    DOLocationID = assigned_order['DOLocationID']
                    new_cluster_id = self.location_to_cluster.get(DOLocationID, None)
                
                    if new_cluster_id is not None:
                        # Update vehicle status to idle
                        self.vehicle_status[i] = 'idle'
                    
                        # Update vehicle's location on the grid
                        old_cluster = self.vehicles_data.at[i, 'Cluster_ID']
                        self.vehicles_data.at[i, 'Cluster_ID'] = new_cluster_id

                        if isinstance(old_cluster, str):
                            old_row, old_col = eval(old_cluster)
                        else:
                            old_row, old_col = old_cluster

                        # Reduce idle vehicle count at the old location
                        if self.idle_vehicle_grid[old_row, old_col] > 0:
                            self.idle_vehicle_grid[old_row, old_col] -= 1

                        # Place the vehicle back on the grid as idle in the new location
                        if isinstance(new_cluster_id, str):
                            new_row, new_col = eval(new_cluster_id)
                        else:
                            new_row, new_col = new_cluster_id

                        self.idle_vehicle_grid[new_row, new_col] += 1

                        # Update the vehicle assignment record with the new drop-off cluster ID
                        self.vehicle_assignments_df.loc[self.vehicle_assignments_df['Vehicle_ID'] == i, 'Dropoff_Cluster_ID'] = new_cluster_id

                        print(f"Vehicle {i} has completed the trip, moved to Cluster {new_cluster_id}, and is now idle.")
                    else:
                        print(f"Warning: DOLocationID {DOLocationID} not mapped to a Cluster_ID.")



    def calculate_reward(self, action, row, col, pending_order_grid):
        alpha = 1.5  # Reward for satisfying excess demand
        beta = 1.0   # Penalty for ignoring demand

        current_demand = pending_order_grid[row, col]
        current_supply = self.idle_vehicle_grid[row, col]
        neighbors = self.get_neighbors(row, col)

        if action == 0:  # Stay in the current cluster
            if current_demand >= current_supply:
                # Reward for satisfying excess demand
                return alpha
            elif all(pending_order_grid[n_row, n_col] <= self.idle_vehicle_grid[n_row, n_col] for n_row, n_col in neighbors):
                if current_demand < current_supply:
                    # No penalty if demand < supply and no neighbors have excess demand
                    return 0
        else:
            if action - 1 < len(neighbors):
                n_row, n_col = neighbors[action - 1]
                neighbor_demand = pending_order_grid[n_row, n_col]
                neighbor_supply = self.idle_vehicle_grid[n_row, n_col]

                if neighbor_demand > neighbor_supply and current_demand < current_supply:
                # Reward for helping neighbors with excess demand
                    return alpha
            if current_demand > current_supply:
                # Penalty for ignoring demand while current cluster has excess demand
                return -beta

        # Default penalty for unaligned actions
        return 0


    
    def train(self, episodes=200, train=True, batch_size=128, validation_interval =5):
        all_shap_values = []
        all_states = []
        
        for episode in range(episodes):
            print(f"\nStarting Episode {episode + 1}/{episodes}")

            self.pending_order_grid = np.random.randint(0, 5, size=self.grid_size)
            total_reward = 0
            total_loss = 0
            step_count = 0
            episode_pending_orders = 0
            episode_rejections = 0
            
            for timestep in range(self.timesteps_per_episode):
                timestep_total_demand = np.sum(self.pending_order_grid)

                # Dispatch orders and update vehicle status
                self.dispatch_orders(self.pending_order_grid, self.taxi_data)
                self.update_vehicle_status(self.timestep_duration)

                # Calculate rejection rate for this timestep
                remaining_pending_orders = np.sum(self.pending_order_grid)
                rejection_rate_timestep = (((remaining_pending_orders * 100) / timestep_total_demand) / timestep_total_demand) * 100 if timestep_total_demand > 0 else 0
                self.rejection_rate_timestep.append(rejection_rate_timestep)

                # Update episode-level rejection stats
                episode_pending_orders += timestep_total_demand
                episode_rejections += remaining_pending_orders

                # Train agent and accumulate losses
                if train:
                    for row in range(self.grid_size[0]):
                        for col in range(self.grid_size[1]):
                            state = self.get_state(row, col, self.pending_order_grid)
                            shap_values = self.agent.explain_action(state)

                            if shap_values is not None:
                                all_shap_values.append(shap_values)
                                all_states.append(state)
                            action = self.agent.act(state)
                            reward = self.calculate_reward(action, row, col, self.pending_order_grid)
                            next_state = self.get_state(row, col, self.pending_order_grid)
                            self.agent.remember(state, action, reward, next_state, done=False)

                            total_reward += reward  # Accumulate reward here

                    loss = self.agent.replay(batch_size)
                    if loss is not None:
                        self.step_losses.append(loss)  # Append training loss
                        total_loss += loss
                        step_count += 1

            # Calculate rejection rate for the episode
            episode_rejection_rate = ((episode_rejections / episode_pending_orders)/episode_pending_orders ) * 100 if episode_pending_orders > 0 else 0
            self.training_rejection_rates.append(episode_rejection_rate)  # Store training rejection rate

            # Record cumulative reward for the episode
            self.training_rewards.append(total_reward)  # Store training reward

            print(f"Episode {episode + 1} - Total Reward: {total_reward}, Average Loss: {total_loss / max(step_count, 1)}")
            print(f"Episode Rejection Rate: {episode_rejection_rate * 100:.2f}%")

            if (episode + 1) % validation_interval == 0:
                print(f"Performing validation after {episode + 1} episodes...")
                val_loss, val_reward, val_rejection_rate = self.validate()
                self.validation_rewards.append(val_reward)
                self.validation_losses.append(val_loss)
                self.validation_rejection_rates.append(val_rejection_rate)
                print(f"Validation after {episode + 1} Episodes - Loss: {val_loss}, Reward: {val_reward}, Rejection Rate: {val_rejection_rate:.2f}%")
            else:
                # Append placeholders for non-validation episodes
                self.validation_rewards.append(None)
                self.validation_losses.append(None)
                self.validation_rejection_rates.append(None)

        
        if all_shap_values and all_states:
            # Convert lists of SHAP values and states to numpy arrays
            all_shap_values = np.vstack(all_shap_values)
            all_states = np.vstack([state[1:] for state in all_states])  # Exclude Cluster_ID from states

            # Check if shap_values has an extra column and remove it if necessary
            if all_shap_values.shape[1] > all_states.shape[1]:
                all_shap_values = all_shap_values[:, :-1]  # Remove the last column

            # Define feature names excluding 'Cluster_ID'
            feature_names = ["Current Demand", "Current Supply"]
            for i in range(1, 9):
                feature_names += [f"Neighbor Demand {i}", f"Neighbor Supply {i}"]

            # Generate and save the bar chart summary plot
            shap.summary_plot(all_shap_values, all_states, feature_names=feature_names, plot_type='bar', show=False)
            plt.savefig("shap_bar_chart_plot.png")
            print("SHAP bar chart saved as shap_bar_chart_plot.png")
        else:
            print("No SHAP values to plot.")


    def validate(self):
        # Initialize metrics
        total_loss = 0
        total_rejections = 0
        total_demand = 0
        total_reward = 0  # Track cumulative reward for validation
        step_count = 0  # To calculate average loss

        # Reset the pending orders grid
        self.pending_order_grid = np.random.randint(0, 5, size=self.grid_size)

        for timestep in range(self.timesteps_per_episode):
            timestep_total_demand = np.sum(self.pending_order_grid)
            total_demand += timestep_total_demand

            # Dispatch orders and update vehicle status
            self.dispatch_orders(self.pending_order_grid, self.val_data)
            self.update_vehicle_status(self.timestep_duration)

            # Calculate reward and loss for validation
            timestep_reward = 0
            for row in range(self.grid_size[0]):
                for col in range(self.grid_size[1]):
                    state = self.get_state(row, col, self.pending_order_grid)
                    action = self.agent.act(state)
                    reward = self.calculate_reward(action, row, col, self.pending_order_grid)
                    timestep_reward += reward
            total_reward += timestep_reward  # Accumulate reward

            # Replay to calculate loss (optional for validation)
            loss = self.agent.replay(batch_size=128)  # Replay even during validation
            if loss is not None:
                total_loss += loss
                step_count += 1

            # Count rejections
            remaining_pending_orders = np.sum(self.pending_order_grid)
            total_rejections += remaining_pending_orders

        # Calculate averages
        avg_loss = total_loss / max(step_count, 1)  # Avoid division by zero
        rejection_rate = ((total_rejections / total_demand) /total_demand) * 100 if total_demand > 0 else 0

        print(f"Validation - Cumulative Reward: {total_reward}, Average Loss: {avg_loss}, Rejection Rate: {rejection_rate:.2f}%")
        return avg_loss, total_reward, rejection_rate  # Return loss, reward, and rejection rate



    
    def test(self, episodes=200):
        print(f"Starting testing phase with {episodes} episodes...")
        self.testing_rewards = []  # Reset cumulative rewards for testing
        self.testing_rejection_rates = []  # Reset rejection rates for testing
    
        for episode in range(episodes):
            print(f"Testing Episode {episode + 1}/{episodes}")

            # Reset idle vehicles grid and pending orders grid
            self.idle_vehicle_grid = np.zeros(self.grid_size)  # Reset idle vehicles
            for position in self.vehicles_data['Cluster_ID']:
                row, col = eval(position)
                self.idle_vehicle_grid[row, col] += 1
        
            self.pending_order_grid = np.random.randint(0, 5, size=self.grid_size)

            # Initialize metrics for the episode
            total_reward = 0
            total_demand = 0
            total_rejections = 0

            for timestep in range(self.timesteps_per_episode):
                # Track total demand for this timestep
                timestep_total_demand = np.sum(self.pending_order_grid)
                total_demand += timestep_total_demand

                # Dispatch orders and update vehicle status
                self.dispatch_orders(self.pending_order_grid, self.test_data)
                self.update_vehicle_status(self.timestep_duration)

                # Calculate remaining pending orders (rejections for this timestep)
                remaining_pending_orders = np.sum(self.pending_order_grid)
                total_rejections += remaining_pending_orders

                # Calculate cumulative reward for this timestep
                timestep_reward = 0
                for row in range(self.grid_size[0]):
                    for col in range(self.grid_size[1]):
                        state = self.get_state(row, col, self.pending_order_grid)
                        action = self.agent.act(state)
                        reward = self.calculate_reward(action, row, col, self.pending_order_grid)
                        timestep_reward += reward

                total_reward += timestep_reward  # Accumulate rewards for the episode

            # Calculate rejection rate for the episode
            rejection_rate = ((total_rejections / total_demand)/ total_demand) * 100 if total_demand > 0 else 0
            self.testing_rejection_rates.append(rejection_rate)  # Store testing rejection rate
            self.testing_rewards.append(total_reward)  # Store cumulative rewards for testing

            print(f"Episode {episode + 1} - Cumulative Reward: {total_reward}, Rejection Rate: {rejection_rate * 100:.2f}%")


    def visualize_idle_vehicles_grid(self, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))
        idle_vehicles_heatmap = ax.imshow(self.idle_vehicle_grid, cmap='Blues', interpolation='nearest', alpha=0.6)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                idle_count = int(self.idle_vehicle_grid[i, j])
                ax.text(j, i, f'{idle_count}', ha='center', va='center', color='black')

        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Idle Vehicles before dispatching from {timestep_start_time} to {timestep_end_time}')

        fig.colorbar(idle_vehicles_heatmap, ax=ax, orientation='vertical', label='Idle Vehicles')
        plt.show()

    def visualize_pending_orders_grid(self, pending_order_grid, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Heatmap for pending orders (red)
        pending_orders_heatmap = ax.imshow(pending_order_grid, cmap='Reds', interpolation='nearest', alpha=0.6)

        # Annotate each cell with the number of pending orders
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                pending_count = int(pending_order_grid[i, j])
                ax.text(j, i, f'{pending_count}', ha='center', va='center', color='black')

        # Set grid labels and titles
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Pending Orders before dispatching from {timestep_start_time} to {timestep_end_time}')

        # Add colorbar
        fig.colorbar(pending_orders_heatmap, ax=ax, orientation='vertical', label='Pending Orders')

        plt.show()
    def visualize_idle_vehicles_grid_after_dispatching(self, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))
        idle_vehicles_heatmap = ax.imshow(self.idle_vehicle_grid, cmap='Blues', interpolation='nearest', alpha=0.6)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                idle_count = int(self.idle_vehicle_grid[i, j])
                ax.text(j, i, f'{idle_count}', ha='center', va='center', color='black')

        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Idle Vehicles after dispatching from {timestep_start_time} to {timestep_end_time}')

        fig.colorbar(idle_vehicles_heatmap, ax=ax, orientation='vertical', label='Idle Vehicles')
        plt.show()

    def visualize_pending_orders_grid_after_dispatching(self, pending_order_grid, timestep_start_time, timestep_end_time):
        fig, ax = plt.subplots(figsize=(10, 10))

        # Heatmap for pending orders (red)
        pending_orders_heatmap = ax.imshow(pending_order_grid, cmap='Reds', interpolation='nearest', alpha=0.6)

        # Annotate each cell with the number of pending orders
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                pending_count = int(pending_order_grid[i, j])
                ax.text(j, i, f'{pending_count}', ha='center', va='center', color='black')

        # Set grid labels and titles
        ax.set_xticks(np.arange(self.grid_size[1]))
        ax.set_yticks(np.arange(self.grid_size[0]))
        ax.set_xticklabels(np.arange(0, self.grid_size[1]))
        ax.set_yticklabels(np.arange(0, self.grid_size[0]))

        ax.set_xlabel('Grid Columns')
        ax.set_ylabel('Grid Rows')
        ax.set_title(f'Pending Orders after dispatching from {timestep_start_time} to {timestep_end_time}')

        # Add colorbar
        fig.colorbar(pending_orders_heatmap, ax=ax, orientation='vertical', label='Pending Orders')

        plt.show()
    def plot_results(self):
        plt.figure(figsize=(15, 5))

        # Plot reward per episode
        plt.subplot(1, 2, 1)
       
        plt.plot(self.rejection_rate_timestep)
        plt.xlabel("Timestep")
        plt.ylabel("Rejection Rate (%)")

        plt.title("Rejection Rate per Timestep")

        # Plot rejection rate per episode
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_rejection_rate )
        plt.xlabel("Episode")
        plt.ylabel("Rejection Rate (%)")
        plt.title("Rejection Rate per Episode")

        plt.tight_layout()
        plt.show()
    
    def plot_training_progress(self):
        # Create a consistent grid with 1 row and 2 columns
        plt.figure(figsize=(15, 5))

        # Plot rewards per episode
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_per_episode, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Reward per Episode")

        # Plot losses per episode
        plt.subplot(1, 2, 2)
        plt.plot(self.losses, label="Loss", color='orange')
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.title("Loss per Episode")

        plt.tight_layout()
        plt.show()

    

    def plot_cumulative_rewards(self, training_rewards, validation_rewards, testing_rewards):
        plt.figure(figsize=(10, 6))

        # Plot training rewards
        plt.plot(training_rewards, label='Training Reward', color='blue')

        # Replace None in validation rewards with the last valid value
        filtered_validation_rewards = []
        last_valid_value = 0  # Start from 0 or a baseline
        for value in validation_rewards:
            if value is not None:
                last_valid_value = value
            filtered_validation_rewards.append(last_valid_value)

        # Plot validation rewards
        plt.plot(filtered_validation_rewards, label='Validation Reward', color='orange')

        # Plot testing rewards
        plt.plot(testing_rewards, label='Testing Reward', color='green')

        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards per Episode')
        plt.legend()
        plt.grid()
        plt.show()


    def plot_rejection_rates(self,training_rejection_rates, validation_rejection_rates, testing_rejection_rates):
        plt.figure(figsize=(10, 6))

        # Tracer les taux de rejet pour l'entraînement
        plt.plot(training_rejection_rates, label='Training Rejection Rate', color='blue')

        # Tracer les taux de rejet pour la validation (sans les valeurs None)
        valid_indices = [i for i, v in enumerate(validation_rejection_rates) if v is not None]
        valid_validation_rejection_rates = [v for v in validation_rejection_rates if v is not None]
        plt.plot(valid_indices, valid_validation_rejection_rates, label='Validation Rejection Rate', color='orange')

        # Tracer les taux de rejet pour le test
        plt.plot(testing_rejection_rates, label='Testing Rejection Rate', color='green')

        # Configuration des axes et légende
        plt.xlabel('Episode')
        plt.ylabel('Rejection Rate')
        plt.title('Rejection Rates per Episode')
        plt.legend()
        plt.grid()
        plt.show()
        
    def plot_losses(self):
        plt.figure(figsize=(10, 6))
    
    # Plot training losses
        plt.plot(self.step_losses, label='Training Loss', color='blue')
    
    # Handle None values in validation losses
        filtered_validation_losses = [loss if loss is not None else 0 for loss in self.validation_losses]
    
    # Plot validation losses
        plt.plot(filtered_validation_losses, label='Validation Loss', color='orange')
    
    # Labels and title
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()



   


def main():
    
   # sys.stdout = open('simulationnew_val1.txt', 'w')
    batch_size = 128
    train_episodes =200
    test_episodes = 200 # Number of test episodes

    print(f"\nTraining with batch size {batch_size}")

    # Initialize the simulation
    sim = Simulation(
        taxi_data_path='F:/new project/v1_30_10/data/taxi_data.csv',
        vehicles_data_path='F:/new project/v1_30_10/vehicle_data.csv',
        grid_size=(20, 20),
        episode_duration=30
    )

    # Train the simulation with the specified parameters
    sim.train(episodes=train_episodes, train=True, batch_size=batch_size, validation_interval =5)
    

    # Test the trained model over a specified number of episodes
    sim.test(episodes=test_episodes)

    # Plot metrics
    #sim.plot_metrics()
    sim.plot_cumulative_rewards(sim.training_rewards, sim.validation_rewards, sim.testing_rewards)

   
    sim.plot_rejection_rates(sim.training_rejection_rates, sim.validation_rejection_rates, sim.testing_rejection_rates)     # Plot rejection rates
    sim.plot_losses()  
    

    # Optionally save results or logs
    print("Simulation complete. Metrics plotted and saved.")
    # Store the results for plotting
    losses_dict = {batch_size: sim.step_losses}
    rewards_train_dict = {batch_size: sim.training_rewards}
    rewards_val_dict = {batch_size: sim.validation_rewards}
    rejection_rate_timestep_dict = {batch_size: sim.rejection_rate_timestep}
    rejection_rate_episode_dict = {batch_size: sim.episode_rejection_rate}

    # Plot the results
    #plot_cumulative_rewards(rewards_train_dict, rewards_val_dict)
    
    #plot_rejection_rate(rejection_rate_timestep_dict, rejection_rate_episode_dict)
    #plot_loss_by_batch_size(losses_dict)



def plot_cumulative_rewards(rewards_train_dict, rewards_val_dict):
    plt.figure(figsize=(10, 6))
    for batch_size, rewards in rewards_train_dict.items():
        plt.plot(rewards, label=f"Batch size {batch_size}")
    for batch_size, rewards in rewards_val_dict.items():
        plt.plot(rewards, label=f"Batch size {batch_size}")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward per Episode for Different Batch Sizes")
    plt.legend()
    plt.grid()
    plt.show()


def plot_rejection_rate(rejection_rate_timestep_dict, rejection_rate_episode_dict):
    plt.figure(figsize=(12, 6))
    
    # Plot per-timestep rejection rate
    plt.subplot(1, 2, 1)
    for batch_size, rejection_rates in rejection_rate_timestep_dict.items():
        plt.plot(rejection_rates, label=f"Batch size {batch_size}")
    plt.xlabel("Timesteps")
    plt.ylabel("Rejection Rate (%)")
    plt.title("Rejection Rate per Timestep")
    plt.legend()
    plt.grid()

    # Plot per-episode rejection rate
    plt.subplot(1, 2, 2)
    for batch_size, rejection_rates in rejection_rate_episode_dict.items():
        plt.plot(rejection_rates, label=f"Batch size {batch_size}")
    plt.xlabel("Episodes")
    plt.ylabel("Rejection Rate (%)")
    plt.title("Rejection Rate per Episode")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def plot_loss_by_batch_size(losses_dict):
    plt.figure(figsize=(10, 6))
    for batch_size, losses in losses_dict.items():
        plt.plot(losses, label=f"Batch size {batch_size}")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss per Training Step for Different Batch Sizes")
    plt.legend()
    plt.grid()
    plt.show()
#sys.stdout.close()
# Call the main function if this script is executed
if __name__ == "__main__":
    main()




  