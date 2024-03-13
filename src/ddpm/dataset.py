import torch as t
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.gan.adversarial_train import get_gan_train_data
from src.utils.filesys import getpath


# filepath = Path(__file__).parent.resolve()
# DATA_PATH = os.path.join(filepath, "levels", "ground", "unique_onehot.npz")
DATA_PATH = getpath('smb/levels')

class MarioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = t.tensor(sample, dtype=t.float32)
        return x

# def load_data(file_path):
#     # data = np.load(file_path)
#     # levels = data['levels']
#     levels = traverse_level_files(DATA_PATH)
#     onehots = []
#     for lvl in levels:
#         num_lvl = lvl.to_num_arr()
#         _, length = num_lvl.shape
#         for s in range(length - W):
#             seg = num_lvl[:, s: s+W]
#             onehot = np.zeros([MarioLevel.n_types, H, W])
#             xs = [seg[i, j] for i, j in product(range(H), range(W))]
#             ys = [k // W for k in range(H * W)]
#             zs = [k % W for k in range(H * W)]
#             onehot[xs, ys, zs] = 1
#             data.append(onehot)
#
#     return [lvl.to_onehot() for lvl in levels]

def create_dataloader(batch_size=32, shuffle=True, num_workers=0):
    # data = load_data(file_path)
    data = get_gan_train_data()
    dataset = MarioDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':
    # mario_dataloader = create_dataloader(DATA_PATH, batch_size=64, shuffle=True)
    # # collect the first batch from mario_dataloader
    # first_batch = next((iter(mario_dataloader)))
    # print(first_batch.shape)
    # # plot all the levels in the first batch
    # fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    # for i, ax in enumerate(axes.flatten()):
    #     level = np.argmax(first_batch[i], axis=0).numpy()
    #     image = get_img_from_level(level)
    #     ax.imshow(255 * np.ones_like(image))  # White background
    #     ax.imshow(image)
    #     ax.axis("off")
    # plt.show()
    data_ = create_dataloader(batch_size=64, shuffle=True).dataset.data
    data_ = np.argmax(data_, axis=1)
    for item in data_:
        print(item.shape)
    # count the occurrence of each number in data
    # for i in range(MarioLevel.n_types):
    #     print(f"{i}: {np.count_nonzero(data_ == i)}")
    print([np.count_nonzero(data_ == i) for i in range(21)])
    pass