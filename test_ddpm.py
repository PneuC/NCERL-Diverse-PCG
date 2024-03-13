import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from src.ddpm.diffusion import Diffusion
from src.ddpm.modules import UNet
# from pytorch_model_summary import summary
# from matplotlib import pyplot as plt
from src.ddpm.dataset import create_dataloader
# from utils.plot import get_img_from_level
from pathlib import Path
# from src.smb.level import MarioLevel
import argparse
import datetime

from src.gan.gankits import process_onehot, get_decoder
from src.smb.level import MarioLevel, lvlhcat, save_batch
from src.utils.filesys import getpath
from src.utils.img import make_img_sheet

# sprite_counts = np.power(np.array([102573, 9114, 1017889, 930, 3032, 7330, 2278, 2279, 5227, 5229, 5419]), 1/4)
sprite_counts = np.power(np.array([
        74977, 15252, 572591, 5826, 1216, 7302, 237, 237, 2852, 1074, 235, 304, 48, 96, 160, 1871, 936, 186, 428, 80, 428
    ]), 1/4
)
min_count = np.min(sprite_counts)

# filepath = Path(__file__).parent.resolve()
# DATA_PATH = os.path.join(filepath, "levels", "ground", "unique_onehot.npz")

def setup_logging(run_name, beta_schedule):
    model_path = os.path.join("models", beta_schedule, run_name)
    result_path = os.path.join("results", beta_schedule, run_name)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    return model_path, result_path

# def plot_images(epoch, sampled_images, result_path):
#     fig = plt.figure(figsize=(30, 15))
#     for i in range(len(sampled_images)):
#         ax1 = fig.add_subplot(4, int(len(sampled_images)/4), i+1)
#         ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
#         level = sampled_images[i].argmax(dim=0).cpu().numpy()
#         level_img = get_img_from_level(level)
#         ax1.imshow(level_img)
#     plt.savefig(os.path.join(result_path, f"{epoch:04d}_sample.png"))
#     plt.close()

# def plot_training_images(epoch, original_img, x_t, noise, predicted_noise, reconstructed_img, training_result_path):
#     fig = plt.figure(figsize=(15, 10))
#     for i in range(2):
#         ax1 = fig.add_subplot(2, 5, i*5+1)
#         ax1.imshow(get_img_from_level(original_img[i].cpu().numpy()))
#         ax1.set_title(f"Original {i}")
#         ax2 = fig.add_subplot(2, 5, i*5+2)
#         ax2.imshow(get_img_from_level(noise[i].cpu().numpy()))
#         ax2.set_title(f"Noise {i}")
#         ax3 = fig.add_subplot(2, 5, i*5+3)
#         ax3.imshow(get_img_from_level(x_t.argmax(dim=1).cpu().numpy()[i]))
#         ax3.set_title(f"x_t {i}")
#         ax4 = fig.add_subplot(2, 5, i*5+4)
#         ax4.imshow(get_img_from_level(predicted_noise[i].cpu().numpy()))
#         ax4.set_title(f"Predicted Noise {i}")
#         ax5 = fig.add_subplot(2, 5, i*5+5)
#         ax5.imshow(get_img_from_level(reconstructed_img.probs.argmax(dim=-1).cpu().numpy()[i]))
#         ax5.set_title(f"Reconstructed Image {i}")
#     plt.savefig(os.path.join(training_result_path, f"{epoch:04d}.png"))
#     plt.close()

def train(args):
    # model_path, result_path = setup_logging(args.run_name, args.beta_schedule)
    # training_result_path = os.path.join(result_path, "training")
    path = getpath(args.res_path)
    os.makedirs(path, exist_ok=True)

    dataloader = create_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = 'cpu' if args.gpuid < 0 else f'cuda:{args.gpuid}'
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(device=device, schedule=args.beta_schedule)
    # logger = SummaryWriter(os.path.join("logs", args.beta_schedule, args.run_name))
    temperatures = torch.tensor(min_count / sprite_counts, dtype=torch.float32).to(device)
    l = len(dataloader)

    # print(summary(model, torch.zeros((64, MarioLevel.n_types, 14, 14)).to(device), diffusion.sample_timesteps(64).to(device), show_input=True))

    # if args.resume_from != 0:
    #     checkpoint = torch.load(os.path.join(model_path, f'ckpt_{args.resume_from}'))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(args.resume_from+1, args.resume_from+args.epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        epoch_loss = {'rec_loss': 0, 'mse': 0, 'loss': 0}
        # pbar = tqdm(dataloader)
        for i, images in enumerate(dataloader):
            images = images.to(device)
            # print(images.shape)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)  # random int from 1~1000
            x_t, noise = diffusion.noise_images(images, t)  # x_t: image with noise at t, noise: gaussian noise
            predicted_noise = model(x_t.float(), t.float()) # returns predicted noise eps_theta

            original_img = images.argmax(dim=1) # batch x 14 x 14
            reconstructed_img = diffusion.sample_only_final(x_t, t, predicted_noise, temperatures)
            rec_loss = -reconstructed_img.log_prob(original_img).sum(dim=(1,2)).mean() # batch
            mse_loss = mse(noise.float(), predicted_noise.float())
            loss = 0.001 * rec_loss + mse_loss
            epoch_loss['rec_loss'] += rec_loss.item()
            epoch_loss['mse'] += mse_loss.item()
            epoch_loss['loss'] += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pbar.set_postfix(LOSS=loss.item())
            # logger.add_scalar("Rec_loss", rec_loss.item(), global_step=(epoch - 1) * l + i)
            # logger.add_scalar("MSE", mse_loss.item(), global_step=(epoch - 1) * l + i)
            # logger.add_scalar("LOSS", loss.item(), global_step=(epoch - 1) * l + i)

        # logger.add_scalar("Epoch_Rec_loss", epoch_loss['rec_loss']/l, global_step=epoch)
        # logger.add_scalar("Epoch_MSE", epoch_loss['mse']/l, global_step=epoch)
        # logger.add_scalar("Epoch_LOSS", epoch_loss['loss']/l, global_step=epoch)
        print(
            '\nIteration: %d' % epoch,
            'rec_loss: %.5g' % (epoch_loss['rec_loss']/l),
            'mse: %.5g' % (epoch_loss['mse']/l)
        )

        # if epoch % 20 == 19:
        #     sampled_images = diffusion.sample(model, n=50)
        #     imgs = [lvl.to_img() for lvl in process_onehot(sampled_images[-1])]
        #     make_img_sheet(imgs, 10, save_path=f'{args.res_path}/sample{epoch+1}.png')

        # plot_images(epoch, sampled_images[-1], result_path)
        # plot_training_images(epoch, original_img, x_t, noise.argmax(dim=1), predicted_noise.argmax(dim=1), reconstructed_img, training_result_path)

        if epoch % 1000 == 0:
            # torch.save(model.state_dict(), os.path.join(model_path, f"ckpt_{epoch:04d}.pt"))
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'Epoch_Rec_loss': epoch_loss['rec_loss']/l,
            #     'Epoch_MSE': epoch_loss['mse']/l,
            #     'Epoch_LOSS': epoch_loss['loss']/l
            # }, getpath(f"{args.res_path}/ddpm_{epoch}.pt"))
            itpath = getpath(path, f'it{epoch}')
            os.makedirs(itpath, exist_ok=True)
            model.save(getpath(path, itpath, 'ddpm.pth'))


    lvls = []
    init_lateves = torch.tensor(np.load(getpath('analysis/initial_seg.npy')))
    gan = get_decoder()
    init_seg_onhots = gan(torch.tensor(init_lateves).view(*init_lateves.shape, 1, 1))
    i = 0
    for init_seg_onehot in init_seg_onhots:
        seg_onehots = diffusion.sample(model, n=25)[-1]
        a = init_seg_onehot.view(1, *init_seg_onehot.shape)
        b = seg_onehots.detach().cpu()
        print(a.shape, b.shape)
        segs = process_onehot(torch.cat([a, b], dim=0))
        level = lvlhcat(segs)
        lvls.append(level)
    save_batch(lvls, getpath(path, 'samples.lvls'))
    model.save(getpath(path, 'ddpm.pth'))

def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    # parser.add_argument("--data_path", type=str, default=DATA_PATH)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--res_path", type=str, default='exp_data/DDPM')
    # parser.add_argument("--image_size", type=int, default=14)
    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta_schedule", type=str, default="quadratic", choices=['linear', 'quadratic', 'sigmoid'])
    parser.add_argument("--run_name", type=str, default=f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    parser.add_argument("--resume_from", type=int, default=0)
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    launch()


