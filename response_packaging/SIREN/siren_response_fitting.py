import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tqdm

response_table = np.load('response/response_38.npy')
print ("response_table", response_table)

print (response_table.shape)
plt.plot(response_table[0,0,:])
plt.savefig('response.png')

# just take the first four curves along the x = 0 line
target = torch.Tensor(response_table[:4,0,:]).to(device)

from model import *
sirenModel = nn.Sequential(Siren(len(target.shape), 128, 3, 1,
                                 outermost_linear = True,
                                 first_omega_0 = 200,
                                 hidden_omega_0 = 200),
                           # final_activation(),
                           )

domain = torch.cartesian_prod(*[torch.arange(dim, dtype = torch.float)
                                # torch.arange(target.shape[1], dtype = torch.float),
                                # torch.arange(target.shape[2], dtype = torch.float),
                                for dim in target.shape]
                              )
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

n_iter = 100000
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

    if not i%10:
        # print ("inf shape", inference.shape)
        curve_model(inference)
        # plt.imshow(inference.detach().numpy().reshape(targetShape))
        colors = ['red', 'green', 'blue', 'orange']
        fig, axes = plt.subplots(4, 1, sharex = True)
        for i, c in enumerate(colors):
            axes[i].plot(target[i,:],
                         color = c)
            axes[i].plot(inference.detach().numpy().reshape(targetShape)[i,:],
                         color = c,
                         ls = '--')
        plt.xlim(3600, 3800)
        plt.savefig('currentInf.png')

        # fig = plt.figure()
        # # inference.backward()
        # grad = gradient(inference, coords)
        # plt.imshow(grad.norm(dim = -1).detach().numpy().reshape(targetShape[:-1]))
        # plt.savefig('currentInfGrad.png')

        # fig = plt.figure()
        # # inference.backward()
        # lapl = laplace(inference, coords)
        # plt.imshow(lapl.detach().numpy().reshape(targetShape[:-1]))
        # plt.savefig('currentInfLapl.png')

        # fig = plt.figure()
        # inference.backward()
        # grad = inference.grad()
        # plt.imshow(grad.detach().numpy().reshape(targetShape))
        # plt.savefig('currentInfGrad.png')
        scheduler.step()

    loss.backward()
    optimizer.step()

