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
    def __init__(self, num_agents, dist_min_thres, num_samples=1000, device="cpu"):
        super(AgentDataset, self).__init__()
        self.num_agents = num_agents
        self.dist_min_thres = dist_min_thres
        self.num_samples = num_samples
        self.device = device
        self.data = [core.generate_data(num_agents, dist_min_thres) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        states, goals = self.data[idx]
        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        goals_tensor = torch.tensor(goals, dtype=torch.float32).to(self.device)
        return states_tensor, goals_tensor

def count_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_list.append(np.mean(acc[acc[:, i] >= 0, i]))
    return acc_list

def train_epoch(num_agents, model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device):
    model_cbf.train()
    model_action.train()
    total_loss = 0

    # Creating the state_gain matrix
    main_diag = torch.eye(2, 4, device=device)
    shifted_diag = torch.zeros(2, 4, device=device)
    shifted_diag[0, 2] = 1
    shifted_diag[1, 3] = 1
    state_gain = main_diag + shifted_diag * np.sqrt(3)

    for j, (states, goals) in enumerate(dataloader):
        loss_lists_np = []
        acc_lists_np = []
        safety_ratios_epoch = []
        safety_ratios_epoch_lqr = []

        # Ensure the correct shapes and data movement
        s_np = states.squeeze(0).to(device)
        g_np = goals.squeeze(0).to(device)
        s_np_lqr = s_np.clone()
        g_np_lqr = g_np.clone()

        optimizer_cbf.zero_grad()
        optimizer_action.zero_grad()

        for i in range(config.INNER_LOOPS):
            a_np = model_action(s_np, g_np)

            if np.random.uniform() < config.ADD_NOISE_PROB:
                noise = torch.normal(0, config.NOISE_SCALE, size=a_np.shape).to(device)
                a_np = a_np + noise

            # Simulating the system for one step
            s_np = s_np + torch.cat([s_np[:, 2:], a_np], dim=1) * config.TIME_STEP

            # Compute safety ratio
            s_np_detached = s_np.detach().cpu().numpy()  # Ensure tensor is detached for numpy operations
            #safety_ratio = 1 - torch.mean(core.ttc_dangerous_mask_np(
            #    s_np_detached, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), dim=1)
            safety_mask_np = core.ttc_dangerous_mask_np(
                s_np_detached, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            
            # Convert the safety mask to a tensor
            safety_mask_tensor = torch.tensor(safety_mask_np, dtype=torch.float32).to(device)
            
            safety_ratio = 1 - torch.mean(safety_mask_tensor, dim=1)
            safety_ratio = torch.mean((safety_ratio == 1).float())
            safety_ratios_epoch.append(safety_ratio.item())

            if torch.mean(torch.norm(s_np[:, :2] - g_np, dim=1)) < config.DIST_MIN_CHECK:
                break

            # CBF and loss calculations
            x = s_np.unsqueeze(1) - s_np.unsqueeze(0)
            h, mask,indices = model_cbf( x, config.DIST_MIN_THRES,indices=None)

            loss_action = core.loss_actions(s=s_np, g=g_np, a=a_np, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            (loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv) = core.loss_derivatives(
                s=s_np, a=a_np, h=h, x=x, r=config.OBS_RADIUS, ttc=config.TIME_TO_COLLISION,
                alpha=config.ALPHA_CBF, time_step=config.TIME_STEP, dist_min_thres=config.DIST_MIN_THRES)

            (loss_dang, loss_safe, acc_dang, acc_safe) = core.loss_barrier(
                h=h, s=s_np, r=config.DIST_MIN_THRES, ttc=config.TIME_TO_COLLISION)

            loss_list_np = [2 * loss_dang, loss_safe, 2 * loss_dang_deriv, loss_safe_deriv, 0.01 * loss_action]
            acc_list_np = [acc_dang, acc_safe, acc_dang_deriv, acc_safe_deriv]
            loss_lists_np.append(loss_list_np)
            acc_lists_np.append(acc_list_np)

        # Running the LQR controller
        for i in range(config.INNER_LOOPS):
            s_ref_lqr = torch.cat([s_np_lqr[:, :2] - g_np_lqr, s_np_lqr[:, 2:]], dim=1)
            a_lqr = -s_ref_lqr.matmul(state_gain.T)
            s_np_lqr = s_np_lqr + torch.cat([s_np_lqr[:, 2:], a_lqr], dim=1) * config.TIME_STEP
            s_np_lqr[:, :2] = torch.clamp(s_np_lqr[:, :2], 0, 1)

            s_np_lqr_detached = s_np_lqr.detach().cpu().numpy()  # Detach for numpy operation
            safety_mask_lqr_np = core.ttc_dangerous_mask_np(
                s_np_lqr_detached, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK)
            
            # Convert the safety mask to a tensor
            safety_mask_lqr_tensor = torch.tensor(safety_mask_lqr_np, dtype=torch.float32).to(device)
            
            safety_ratio_lqr = 1 - torch.mean(safety_mask_lqr_tensor, dim=1)
            #safety_ratio_lqr = 1 - torch.mean(core.ttc_dangerous_mask_np(
            #    s_np_lqr_detached, config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK), axis=1)
            safety_ratio_lqr = torch.mean((safety_ratio_lqr == 1).float())
            safety_ratios_epoch_lqr.append(safety_ratio_lqr.item())

            if torch.mean(torch.norm(s_np_lqr[:, :2] - g_np_lqr, dim=1)) < config.DIST_MIN_CHECK:
                break

        # Combining losses
        params = list(model_action.parameters()) + list(model_cbf.parameters())
        weight_loss = config.WEIGHT_DECAY * sum([param.pow(2).sum() for param in params])
        loss = 10 * (torch.sum(torch.stack([tensor for sublist in loss_lists_np for tensor in sublist])) + weight_loss)

        # Backpropagation
        loss.backward()
        optimizer_cbf.step()
        optimizer_action.step()

        total_loss += loss.item()
        avg_loss = total_loss / (j + 1)

        # Saving the model at specific intervals
        if not os.path.exists('models'):
            os.makedirs('models')

        # Saving the model at specific intervals
        if (j + 1) % config.SAVE_STEPS == 0:
            torch.save({
                'model_cbf_state_dict': model_cbf.state_dict(),
                'model_action_state_dict': model_action.state_dict(),
            }, os.path.join('models', f'model_epoch_{j + 1}.pth'))
        print(f'Epoch [{j + 1}/{config.TRAIN_STEPS}], Loss: {avg_loss:.4f}, accuracy: {np.array(count_accuracy(acc_lists_np))}')

    return avg_loss

def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model_cbf = core.NetworkCBF().to(device)
    model_action = core.NetworkAction().to(device)
    
    optimizer_cbf = optim.SGD(model_cbf.parameters(), lr=config.LEARNING_RATE)
    optimizer_action = optim.SGD(model_action.parameters(), lr=config.LEARNING_RATE)

    dataset = AgentDataset(args.num_agents, config.DIST_MIN_THRES, num_samples=10000, device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    avg_loss = train_epoch(args.num_agents, model_cbf, model_action, dataloader, optimizer_cbf, optimizer_action, device)
    print(f'Final loss {avg_loss}')

if __name__ == '__main__':
    main()
