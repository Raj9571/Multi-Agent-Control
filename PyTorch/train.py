import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class AgentDataset(Dataset):
    def __init__(self, num_agents, dist_min_thres, num_samples=1000):
        super(AgentDataset, self).__init__()
        self.num_agents = num_agents
        self.dist_min_thres = dist_min_thres
        self.num_samples = num_samples
        # Generate data upfront for simplicity; consider generating on-the-fly for large datasets
        self.data = [core.generate_data(num_agents, dist_min_thres) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        states, goals = self.data[idx]
        return torch.tensor(states, dtype=torch.float32), torch.tensor(goals, dtype=torch.float32)
        

#single epoch loop
def train_epoch(model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device):
    model_cbf.train()
    model_action.train()
    total_loss = 0
    for states, goals in dataloader:
        states, goals = states.to(device), goals.to(device)

        optimizer_cbf.zero_grad()
        optimizer_action.zero_grad()

        h, mask = model_cbf(states, config.DIST_MIN_THRES)
        actions = model_action(states, goals)
        loss_action = core.loss_actions(
        s=s, g=g, a=a, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

        x = torch.unsqueeze(s, 1) - torch.unsqueeze(s, 0)
        
        (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
        ) = core.loss_derivatives(s=states, a=actions, h=h, x=x, r=config.DIST_MIN_THRES, 
        indices=indices, ttc=config.TIME_TO_COLLISION, alpha=config.ALPHA_CBF)

        (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
        h=h, s=states, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION, indices=indices)

        loss_list = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action]
        acc_list = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
    
        WEIGHT_DECAY = config.WEIGHT_DECAY

        # Getting the list of trainable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        
        # Calculating the weight decay loss
        weight_loss = [WEIGHT_DECAY * (param.pow(2).sum()) for param in params]
        
        # Calculating the total loss
        loss = 10 * (sum(loss_list) + sum(weight_loss))
        
        #Need to define a different function for loss_fucntions, as there are 3 losses we need to calculate
        loss.backward()

        optimizer_cbf.step()
        optimizer_action.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
    
#main
def main():
    args = parse_args()

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cbf = NetworkCBF().to(device)
    model_action = NetworkAction().to(device)
    
    optimizer_cbf = optim.Adam(model_cbf.parameters(), lr=config.LEARNING_RATE)
    optimizer_action = optim.Adam(model_action.parameters(), lr=config.LEARNING_RATE)

    dataset = AgentDataset(args.num_agents, config.DIST_MIN_THRES, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(config.TRAIN_STEPS):
        avg_loss = train_epoch(model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device)
        print(f'Epoch [{epoch+1}/{config.TRAIN_STEPS}], Loss: {avg_loss:.4f}')

        if (epoch + 1) % config.SAVE_STEPS == 0:
            torch.save({
                'model_cbf_state_dict': model_cbf.state_dict(),
                'model_action_state_dict': model_action.state_dict(),
            }, os.path.join('models', f'model_epoch_{epoch+1}.pth'))
            print(f'Models saved at epoch {epoch+1}')
            
if __name__ == '__main__':
    main()

