#class of CBF
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
         x = F.relu(self.conv1(x.permute(0,2,1))
         x = F.relu(self.conv2(x))
         x = F.relu(self.conv3(x))
         x = self.conv4(x)
         # Apply mask
         x = x * mask

         return x, mask

#class of action
class NetworkAction(nn.Module):
    def __init__(self, top_k, obs_radius=1.0):
        super(NetworkAction, self).__init__()
        self.top_k = top_k
        self.obs_radius = obs_radius
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=4)

    def forward(self, s, g):
        batch_size, seq_len, _ = s.shape
        #eye = torch.eye(seq_len, device=s.device).expand(batch_size, -1, -1).unsqueeze(1)
        x = torch.unsqueeze(s, 1) - torch.unsqueeze(s, 0)  # NxNx4
        x = torch.cat([x, torch.eye(x.size(0), device=x.device).unsqueeze(2)], dim=2)
    
        # Filter out distant agents and adjust input for convolution layers
        x, _ = self.remove_distant_agents(x)
        #x = x.transpose(1, 2)  # BxCxN
        
        # Apply convolution layers
        x = F.relu(self.conv1(x.permute(0,2,1))
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

def remove_distant_agents(x, indices=None):
    n, _, c = x.size()
    if n <= config.TOP_K:
        return x, False
    d_norm = torch.sqrt(torch.sum(torch.square(x[:, :, :2]) + 1e-6, dim=2))
    if indices is not None:
        x = x[indices]
        return x, indices
    _, indices = torch.topk(-d_norm, k = config.TOP_K, dim=1)
    row_indices = torch.arange(indices.size(0), device=indices.device).unsqueeze(1).expand(-1, config.TOP_K)
    row_indices = row_indices.reshape(-1, 1)
    column_indices = indices.reshape(-1, 1)
    indices = torch.cat([row_indices, column_indices], dim=1)
    x = x[indices[:, 0], indices[:, 1], :]
    x = x.reshape(n, config.TOP_K, c)
    return x, indices
