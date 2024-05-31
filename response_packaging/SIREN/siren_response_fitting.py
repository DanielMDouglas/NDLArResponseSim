import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tqdm

response_table = np.load('../../response_tables/response_38.npy')
print ("response_table", response_table)

print (response_table.shape)
plt.plot(response_table[0,0,:])
plt.savefig('response.png')

# just take the first four curves along the x = 0 line
target = torch.Tensor(response_table[:5,:5,:]).to(device)

from model import *
print (target.shape, len(target.shape))
sirenModel = nn.Sequential(Siren(len(target.shape), 128, 4, 1,
                                 outermost_linear = True,
                                 # first_omega_0 = 200,
                                 # hidden_omega_0 = 200,
                                 first_omega_0 = 15,
                                 hidden_omega_0 = 15,
),
                           # final_activation(),
                           ).to(device)

domain = torch.cartesian_prod(*[torch.arange(dim, dtype = torch.float)
                                # torch.arange(target.shape[1], dtype = torch.float),
                                # torch.arange(target.shape[2], dtype = torch.float),
                                for dim in target.shape]
                              ).to(device)
domain /= np.max(target.shape)
domain *= 2
domain -= 1
coords = domain
# print(torch.meshgrid(torch.arange(target.shape[0]),
#                      torch.arange(target.shape[1]),
#                      ))

# plt.figure()
# plt.imshow(target)
targetShape = target.shape
# plt.savefig('targetImage.png')

# def curve_model(inference):
#     shaped_inference = inference.reshape(targetShape)
#     print ("reshaped", shaped_inference.shape)

    # cutoff_response = torch.where(torch.arange(targetShape[1]) > cutoff),
    # shaped_inference
    
    # return cutoff_response

optimizer = torch.optim.Adam(sirenModel.parameters(),
                             lr = 1.e-6)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

n_iter = 1000000
pbar = tqdm.tqdm(range(n_iter))
for i in pbar:
    inference, coords = sirenModel(domain)
    # inference = sirenModel(domain)
    # print (target.shape)
    # print (inference.shape)
    loss = nn.MSELoss()(inference.flatten(), target.flatten())

    pbarMessage = " ".join(["loss:",
                            str(round(loss.item(), 4))])
    pbar.set_description(pbarMessage)

    if not i%1000:
        # print ("inf shape", inference.shape)
        # print ("target shape", targetShape)
        # curve_model(inference)
        # plt.imshow(inference.detach().numpy().reshape(targetShape))
        colors = 5*['red', 'green', 'blue', 'orange', 'grey']
        current_fig, current_axes = plt.subplots(5*5, 1,
                                                 sharex = True,
                                                 figsize = [6.4, 9.6])
        current_fig.subplots_adjust(hspace = 0)
        diff_fig, diff_axes = plt.subplots(5*5, 1,
                                           sharex = True,
                                           figsize = [6.4, 9.6])
        diff_fig.subplots_adjust(hspace = 0)
        for i, c in enumerate(colors):
            # indexTuple = ((i//4)-1, i%4, :)
            current_axes[i].plot(target[(i//5)-1,i%5,:].cpu().numpy(),
                                 color = c)
            current_axes[i].plot(inference.detach().cpu().numpy().reshape(targetShape)[(i//5)-1,i%5,:],
                                 color = c,
                                 ls = '--')
            current_axes[i].set_xlim(3725, 3800)
            current_axes[i].set_ylim(-0.2, 6)

            diff_axes[i].plot(target[(i//5)-1,i%5,:].cpu().numpy() -\
                              inference.detach().cpu().numpy().reshape(targetShape)[(i//5)-1,i%5,:],
                              color = c)
            diff_axes[i].axhline(y = 0,
                                 ls = '--',
                                 color = c)
            diff_axes[i].set_xlim(3725, 3800)
            diff_axes[i].set_ylim(-0.5, 0.5)

            if not i == 15:
                current_axes[i].tick_params(bottom = False)
                diff_axes[i].tick_params(bottom = False)
                current_axes[i].set_yticklabels([])
                diff_axes[i].set_yticklabels([])
                
        current_fig.savefig('currentInf.png')
        diff_fig.savefig('diff.png')

        scheduler.step()

    loss.backward()
    optimizer.step()

