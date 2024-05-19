import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import core
import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args

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
def train_epoch(num_agents, model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device):
    model_cbf.train()
    model_action.train()
    total_loss = 0

    state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
    j=0
    for states, goals in dataloader:
        loss_lists_np = []
        acc_lists_np = []
        safety_ratios_epoch = []
        safety_ratios_epoch_lqr = []
        s_np = states
        g_np = goals
        s_np_lqr = s_np
        g_np_lqr = g_np
        optimizer_cbf.zero_grad()
        optimizer_action.zero_grad()
        


        for i in range(config.INNER_LOOPS):
            
            a_np = model_action(s_np, g_np)
            if np.random.uniform() < config.ADD_NOISE_PROB:
                noise = np.random.normal(size=np.shape(a_np)) * config.NOISE_SCALE
                a_np = a_np + noise
            
            s_np = s_np + np.concatenate([s_np[:,2:], a_np.detach().numpy()], axis=1)*config.TIME_STEP
            safety_ratio = 1 - np.mean(core.ttc_dangerous_mask_np(
            s_np, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            safety_ratio = np.mean(safety_ratio == 1)
            safety_ratios_epoch.append(safety_ratio)
            if np.mean(
                    np.linalg.norm(s_np[:, :2] - g_np, axis=1)
                    ) < config.DIST_MIN_CHECK:
                    break
            x = torch.unsqueeze(s_np, 1) - torch.unsqueeze(s_np, 0)
            h, mask = model_cbf(x, config.DIST_MIN_THRES)
            loss_action = core.loss_actions(
            s=s_np, g=g_np, a=a_np, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            x = torch.unsqueeze(s_np, 1) - torch.unsqueeze(s_np, 0)
            
            (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv
            ) = core.loss_derivatives(s=s_np, a=a_np, h=h, x=x, r=config.OBS_RADIUS, ttc=config.TIME_TO_COLLISION, alpha=config.ALPHA_CBF, time_step = config.TIME_STEP, dist_min_thres = config.DIST_MIN_THRES)

            (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
            h=h, s=s_np, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            loss_list_np = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action]
            acc_list_np = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
            loss_lists_np.append(loss_list_np)
            acc_lists_np.append(acc_list_np)

        # run the system with the LQR controller without collision avoidance as the baseline
        for i in range(config.INNER_LOOPS):
            state_gain = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
            s_ref_lqr = np.concatenate([s_np_lqr[:, :2] - g_np_lqr, s_np_lqr[:, 2:]], axis=1)
            a_lqr = -s_ref_lqr.dot(state_gain.T)
            s_np_lqr = s_np_lqr + np.concatenate([s_np_lqr[:, 2:], a_lqr], axis=1) * config.TIME_STEP
            s_np_lqr[:, :2] = np.clip(s_np_lqr[:, :2], 0, 1)
            safety_ratio_lqr = 1 - np.mean(core.ttc_dangerous_mask_np(
                s_np_lqr, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            safety_ratio = np.mean(safety_ratio == 1)
            safety_ratios_epoch_lqr.append(safety_ratio_lqr)

            if np.mean(
                np.linalg.norm(s_np_lqr[:, :2] - g_np_lqr, axis=1)
                ) < config.DIST_MIN_CHECK:
                break

       
        WEIGHT_DECAY = config.WEIGHT_DECAY

        # Getting the list of trainable parameters
        params = [p for p in model_action.parameters() if p.requires_grad]
        
        # Calculating the weight decay loss
        weight_loss = [WEIGHT_DECAY * (param.pow(2).sum()) for param in params]
        # Calculating the total loss
        loss = 10 * (torch.sum(torch.stack([tensor for sublist in loss_lists_np for tensor in sublist])))# + sum(weight_loss))
        
        loss.backward()

        optimizer_cbf.step()
        optimizer_action.step()

        total_loss += loss.item()
        avg_loss = total_loss /(j+1)
        j = j + 1
        print(f'Epoch [{j+1}/{config.TRAIN_STEPS}], Loss: {avg_loss:.4f}')

    return avg_loss
    
#main
def main():
    args = parse_args()
    if args.gpu and torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model_cbf = core.NetworkCBF().to(device)
    model_action = core.NetworkAction().to(device)
    
    optimizer_cbf = optim.Adam(model_cbf.parameters(), lr=config.LEARNING_RATE)
    optimizer_action = optim.Adam(model_action.parameters(), lr=config.LEARNING_RATE)

    dataloader = AgentDataset(args.num_agents, config.DIST_MIN_THRES, num_samples=1000)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(config.TRAIN_STEPS):
        avg_loss = train_epoch(args.num_agents, model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device)
        
        
        if (epoch + 1) % config.SAVE_STEPS == 0:
            torch.save({
                'model_cbf_state_dict': model_cbf.state_dict(),
                'model_action_state_dict': model_action.state_dict(),
            }, os.path.join('models', f'model_epoch_{epoch+1}.pth'))
            print(f'Models saved at epoch {epoch+1}')
            
if __name__ == '__main__':
    main()

