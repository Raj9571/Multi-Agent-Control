import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


import core
import config

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()

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

        loss = core.loss_functions(h, actions, states, goals, mask)  # Custom loss computation
        loss.backward()

        optimizer_cbf.step()
        optimizer_action.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    args = parse_args()

    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cbf = core.NetworkCBF(config.OBS_RADIUS).to(device)
    model_action = core.NetworkAction(config.TOP_K, config.OBS_RADIUS).to(device)

    optimizer_cbf = optim.Adam(model_cbf.parameters(), lr=config.LEARNING_RATE)
    optimizer_action = optim.Adam(model_action.parameters(), lr=config.LEARNING_RATE)

    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model_cbf.load_state_dict(checkpoint['model_cbf_state_dict'])
        model_action.load_state_dict(checkpoint['model_action_state_dict'])

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

