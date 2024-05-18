import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, np.pi*2, num=num, endpoint=False).reshape(-1, 1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    return circle


def generate_obstacle_rectangle(center, sides, num=12):
    # calculate the number of points on each side of the rectangle
    a, b = sides # side lengths
    n_side_1 = int(num // 2 * a / (a+b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3
    # top
    side_1 = np.concatenate([
        np.linspace(-a/2, a/2, n_side_1, endpoint=False).reshape(-1, 1), 
        b/2 * np.ones(n_side_1).reshape(-1, 1)], axis=1)
    # right
    side_2 = np.concatenate([
        a/2 * np.ones(n_side_2).reshape(-1, 1),
        np.linspace(b/2, -b/2, n_side_2, endpoint=False).reshape(-1, 1)], axis=1)
    # bottom
    side_3 = np.concatenate([
        np.linspace(a/2, -a/2, n_side_3, endpoint=False).reshape(-1, 1), 
        -b/2 * np.ones(n_side_3).reshape(-1, 1)], axis=1)
    # left
    side_4 = np.concatenate([
        -a/2 * np.ones(n_side_4).reshape(-1, 1),
        np.linspace(-b/2, b/2, n_side_4, endpoint=False).reshape(-1, 1)], axis=1)

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)
    return rectangle


def generate_data(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros(shape=(num_agents, 2), dtype=np.float32)
    goals = np.zeros(shape=(num_agents, 2), dtype=np.float32)

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(size=(2,)) * side_length
        dist_min = np.linalg.norm(states - candidate, axis=1).min()
        if dist_min <= dist_min_thres:
            continue
        states[i] = candidate
        i = i + 1

    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i]
        dist_min = np.linalg.norm(goals - candidate, axis=1).min()
        if dist_min <= dist_min_thres:
            continue
        goals[i] = candidate
        i = i + 1

    states = np.concatenate(
        [states, np.zeros(shape=(num_agents, 2), dtype=np.float32)], axis=1)
    return states, goals


class NetworkCBF(nn.Module):
    def __init__(self, obs_radius):
        super(NetworkCBF, self).__init__()
        self.obs_radius = obs_radius
        # Adjust in_channels to match the dimension after concatenation
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

    
    def forward(self, x, r):
         # Calculate norm
         d_norm = torch.sqrt(torch.sum(x[:, :, :2]**2 + 1e-4, dim=2))
         # Identity matrix for self identification
         eye = torch.eye(x.shape[0], device=x.device).unsqueeze(-1)  # Ensure correct dimension
         # Concatenate additional features
         x = torch.cat([x, eye, (d_norm.unsqueeze(-1) - r)], dim=-1)  # Use dim=-1 for last dimension
    
         # (Optional) Implement and apply remove_distant_agents logic here
         # x = remove_distant_agents(x, k=self.top_k, device=x.device) if needed

         # Ensure the distance calculation and masking logic align with your new tensor shape
         dist = torch.sqrt(torch.sum(x[..., :2]**2 + 1e-4, dim=-1, keepdim=True))
         mask = (dist <= self.obs_radius).float()

         # Pass through convolutional layers
         x = F.relu(self.conv1(x))
         x = F.relu(self.conv2(x))
         x = F.relu(self.conv3(x))
         x = self.conv4(x)
         # Apply mask
         x = x * mask

         return x, mask

network_cbf = NetworkCBF(obs_radius=config.OBS_RADIUS)

class NetworkAction(nn.Module):
    def __init__(self, top_k, obs_radius=1.0):
        super(NetworkAction, self).__init__()
        self.top_k = top_k
        self.obs_radius = obs_radius
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=4)

    def forward(self, s, g):
        batch_size, seq_len, _ = s.shape
        eye = torch.eye(seq_len, device=s.device).expand(batch_size, -1, -1).unsqueeze(1)
        x = s.unsqueeze(2) - s.unsqueeze(1)  # NxNx4
        x = torch.cat([x, eye], dim=3)  # Add eye matrix for identifying the agent itself

        # Filter out distant agents and adjust input for convolution layers
        x, _ = self.remove_distant_agents(x)
        x = x.transpose(1, 2)  # BxCxN
        
        # Apply convolution layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Apply global max pooling
        x, _ = torch.max(x, dim=2)
        
        # Combine with goal and current velocity information
        x = torch.cat([x, s[:, :, :2] - g, s[:, :, 2:]], dim=1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        x = 2.0 * torch.sigmoid(x) - 1.0
        return x

    def remove_distant_agents(self, x):
        batch_size, seq_len, _, _ = x.shape
        dists = torch.norm(x[..., :2], dim=3)  # Compute distances based on position differences
        dists, indices = dists.topk(self.top_k, dim=2, largest=False)  # Find top_k closest agents

        # Gather the top_k entries for each element in the batch
        x_top_k = torch.gather(x, 2, indices.unsqueeze(-1).expand(-1, -1, -1, x.size(3)))

        # Create a mask to identify top_k entries
        mask = torch.zeros(batch_size, seq_len, device=x.device).scatter_(1, indices, 1).unsqueeze(2)

        return x_top_k, mask


def dynamics(s, a):
    """
    The ground robot dynamics in PyTorch.

    Args:
        s (Tensor): The current state, shape (N, 4).
        a (Tensor): The acceleration taken by each agent, shape (N, 2).

    Returns:
        Tensor: The time derivative of s, shape (N, 4).
    """
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt



def loss_barrier(h, s, r, ttc, eps=[1e-3, 0]):
    """
    Build the loss function for the control barrier functions in PyTorch.

    Args:
        h (Tensor): The control barrier function, shape (N, N, 1).
        s (Tensor): The current state of N agents, shape (N, 4).
        r (float): The radius of the safe regions.
        ttc (float): The threshold of time to collision.
    
    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: The dangerous loss, safe loss, 
        accuracy of dangerous conditions, and accuracy of safe conditions.
    """
    h_reshape = h.view(-1)
    dang_mask = ttc_dangerous_mask(s, r=r, ttc=ttc)  # Assuming adaptation to PyTorch
    dang_mask_reshape = dang_mask.view(-1)
    safe_mask_reshape = ~dang_mask_reshape

    dang_h = h_reshape[dang_mask_reshape]
    safe_h = h_reshape[safe_mask_reshape]

    num_dang = dang_h.size(0)
    num_safe = safe_h.size(0)

    loss_dang = torch.sum(torch.maximum(dang_h + eps[0], torch.tensor(0.))) / (1e-5 + num_dang)
    loss_safe = torch.sum(torch.maximum(-safe_h + eps[1], torch.tensor(0.))) / (1e-5 + num_safe)

    acc_dang = torch.sum((dang_h <= 0).float()) / (1e-5 + num_dang)
    acc_safe = torch.sum((safe_h > 0).float()) / (1e-5 + num_safe)

    acc_dang = acc_dang if num_dang > 0 else torch.tensor(-1.0)
    acc_safe = acc_safe if num_safe > 0 else torch.tensor(-1.0)

    return loss_dang, loss_safe, acc_dang, acc_safe


def loss_derivatives(s, a, h, x, r, ttc, alpha, time_step, dist_min_thres, eps=[1e-3, 0]):
    """
    Calculate the loss based on derivatives of the control barrier function in PyTorch.

    Args:
        s (Tensor): The current state of N agents, shape (N, 4).
        a (Tensor): The acceleration taken by each agent, shape (N, 2).
        h (Tensor): The control barrier function, shape (N, N, 1).
        x (Tensor): Input features for the network_cbf, possibly the relative positions.
        r (float): The radius of the safe regions.
        ttc (float): The threshold of time to collision.
        alpha (float): Scaling factor for the derivative term.
        time_step (float): Time step used for the derivative approximation.
        dist_min_thres (float): Minimum distance threshold for considering agents.
        eps (list): Epsilon values for loss calculations.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: The dangerous derivative loss, safe derivative loss,
                                               accuracy of dangerous conditions, accuracy of safe conditions.
    """
    dsdt = dynamics(s, a)
    s_next = s + dsdt * time_step

    x_next = torch.unsqueeze(s_next, 1) - torch.unsqueeze(s_next, 0)
    h_next, mask_next, _ = network_cbf(x=x_next, r=dist_min_thres)  # Assuming adaptation to PyTorch

    deriv = h_next - h + time_step * alpha * h

    deriv_reshape = deriv.view(-1)
    dang_mask = ttc_dangerous_mask(s=s, r=r, ttc=ttc)  # Assuming adaptation to PyTorch
    dang_mask_reshape = dang_mask.view(-1)
    safe_mask_reshape = ~dang_mask_reshape

    dang_deriv = deriv_reshape[dang_mask_reshape]
    safe_deriv = deriv_reshape[safe_mask_reshape]

    num_dang = dang_deriv.size(0)
    num_safe = safe_deriv.size(0)

    loss_dang_deriv = torch.sum(torch.maximum(-dang_deriv + eps[0], torch.tensor(0.))) / (1e-5 + num_dang)
    loss_safe_deriv = torch.sum(torch.maximum(-safe_deriv + eps[1], torch.tensor(0.))) / (1e-5 + num_safe)

    acc_dang_deriv = torch.sum((dang_deriv >= 0).float()) / (1e-5 + num_dang)
    acc_safe_deriv = torch.sum((safe_deriv >= 0).float()) / (1e-5 + num_safe)

    acc_dang_deriv = acc_dang_deriv if num_dang > 0 else torch.tensor(-1.0)
    acc_safe_deriv = acc_safe_deriv if num_safe > 0 else torch.tensor(-1.0)

    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv

def loss_actions(s, g, a, r, ttc):
    """
    Calculate action loss in PyTorch.

    Args:
        s (Tensor): The current state of N agents, shape (N, 4).
        g (Tensor): The goal state of N agents, shape (N, 2).
        a (Tensor): The action taken by each agent, shape (N, 2).
        r (float): The radius of the safe regions. (Not used in this function, but kept for consistency)
        ttc (float): The threshold of time to collision. (Not used in this function, but kept for consistency)

    Returns:
        Tensor: The mean of the absolute difference in norms between the reference and actual actions.
    """
    # Create the state_gain matrix in PyTorch
    state_gain = -torch.tensor(
        np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32)

    # Concatenate the relative positions and velocities
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    
    # Matrix multiplication in PyTorch
    action_ref = torch.matmul(s_ref, state_gain.T)  # Transpose state_gain for correct dimensionality
    
    # Calculate the norms of the reference actions and the actual actions
    action_ref_norm = torch.sum(action_ref ** 2, dim=1)
    action_net_norm = torch.sum(a ** 2, dim=1)
    
    # Calculate the absolute difference in norms and then the mean loss
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)

    return loss


def statics(s, a, h, alpha, time_step, dist_min_thres):
    
    dsdt = dynamics(s, a)  # Assuming adaptation to PyTorch
    s_next = s + dsdt * time_step

    x_next = torch.unsqueeze(s_next, 1) - torch.unsqueeze(s_next, 0)
    h_next, mask_next, _ = network_cbf(x=x_next, r=dist_min_thres)  # Assuming adaptation to PyTorch

    deriv = h_next - h + time_step * alpha * h

    mean_deriv = torch.mean(deriv)
    std_deriv = torch.sqrt(torch.mean((deriv - mean_deriv) ** 2))
    prob_neg = torch.mean((deriv < 0).float())

    return mean_deriv, std_deriv, prob_neg


def ttc_dangerous_mask(s, r, ttc, top_k):
    """
    Calculate a mask identifying dangerous situations based on time-to-collision (TTC) in PyTorch.

    Args:
        s (Tensor): The current state of N agents, shape (N, 4), where each row is [x, y, vx, vy].
        r (float): The radius of the safe regions.
        ttc (float): The threshold of time to collision.
        top_k (int): The top K nearest agents to consider for collision checking.

    Returns:
        Tensor: A boolean tensor indicating dangerous situations, shape (N, N, 1).
    """
    # Assuming remove_distant_agents is adapted to PyTorch and returns appropriate tensors
    s_diff = torch.unsqueeze(s, 1) - torch.unsqueeze(s, 0)
    s_diff = torch.cat([s_diff, torch.unsqueeze(torch.eye(s.size(0), device=s.device), 2)], dim=2)
    s_diff, _ = remove_distant_agents(s_diff, top_k)  # Placeholder for actual implementation
    
    x, y, vx, vy, eye = torch.split(s_diff, [1, 1, 1, 1, 1], dim=2)
    x = x + eye
    y = y + eye
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2
    dist_dangerous = gamma < 0

    has_two_positive_roots = ((beta ** 2 - 4 * alpha * gamma) > 0) & (gamma > 0) & (beta < 0)
    root_less_than_ttc = ((-beta - 2 * alpha * ttc) < 0) | (((beta + 2 * alpha * ttc) ** 2) < (beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc
    ttc_dangerous = dist_dangerous | has_root_less_than_ttc

    return ttc_dangerous.unsqueeze(-1)  # Adding the last dimension to match TensorFlow's shape


def ttc_dangerous_mask_np(s, r, ttc):
    s_diff = np.expand_dims(s, 1) - np.expand_dims(s, 0)
    x, y, vx, vy = np.split(s_diff, 4, axis=2)
    x = x + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    y = y + np.expand_dims(np.eye(np.shape(s)[0]), 2)
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2
    dist_dangerous = np.less(gamma, 0)

    has_two_positive_roots = np.logical_and(
        np.greater(beta ** 2 - 4 * alpha * gamma, 0),
        np.logical_and(np.greater(gamma, 0), np.less(beta, 0)))
    root_less_than_ttc = np.logical_or(
        np.less(-beta - 2 * alpha * ttc, 0),
        np.less((beta + 2 * alpha * ttc) ** 2, beta ** 2 - 4 * alpha * gamma))
    has_root_less_than_ttc = np.logical_and(has_two_positive_roots, root_less_than_ttc)
    ttc_dangerous = np.logical_or(dist_dangerous, has_root_less_than_ttc)

    return ttc_dangerous


def remove_distant_agents(x, k, indices=None):
    n, _, c = x.size()
    if n <= k:
        return x, False
    
    d_norm = torch.sqrt(torch.sum(torch.square(x[:, :, :2]) + 1e-6, dim=2))
    
    if indices is not None:
        x = x[indices.flatten()].reshape(n, k, c)
        return x, indices
    
    _, indices = torch.topk(-d_norm, k=k, dim=1)
    row_indices = torch.arange(indices.size(0)).unsqueeze(1).expand_as(indices)
    row_indices = row_indices.flatten().unsqueeze(1)
    column_indices = indices.flatten().unsqueeze(1)
    indices = torch.cat((row_indices, column_indices), dim=1)
    x = x[indices.flatten()].reshape(n, k, c)
    
    return x, indices