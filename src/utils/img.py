import os
from math import ceil
from src.utils.filesys import getpath
from PIL import Image

def make_img_sheet(imgs, ncols, x_margin=6, y_margin=6, save_path='./image.png', create_folder=True):
    nrows = ceil(len(imgs) / ncols)

    w, h = imgs[0].width, imgs[0].height
    w_canvas = (w + x_margin) * ncols - x_margin
    h_canvas = (h + y_margin) * nrows - y_margin
    canvas = Image.new('RGBA', (w_canvas, h_canvas), (0, 0, 0, 0))
    # canvas.fill(margin_color)
    for i in range(len(imgs)):
        row_id, col_id = i // ncols, i % ncols
        canvas.paste(imgs[i], ((w + x_margin) * col_id, (h + y_margin) * row_id), imgs[i])

    if not save_path is None:
        if create_folder: os.makedirs(os.path.split(getpath(save_path))[0], exist_ok=True)
        canvas.save(getpath(save_path))
    return canvas
