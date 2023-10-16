import tensorflow as tf
from tensorflow import keras

import numpy as np
import os
import glob
import argparse
from PIL import Image
import imageio
from skimage import img_as_ubyte

from model.model import resnet_model, bce_dice_loss

DIM_W = 501
DIM_H = 901


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image", type=str, required=False, help="Path to single tif image"
    )
    parser.add_argument(
        "--folder", type=str, required=False, help="Path to folder of tif images"
    )
    parser.add_argument(
        "--output_path", type=str, required=False, help="Path to output folder"
    )

    return parser.parse_args()


def build_model(DIM_H, DIM_W):
    model = resnet_model(DIM_H, DIM_W)

    model.compile(
        optimizer=keras.optimizers.Adam(lr=3e-4),
        loss=bce_dice_loss,
        metrics=["accuracy"],
    )

    print(model.summary())

    model.load_weights("model/checkpoint.h5")

    return model


def predict_full_image(model, image, output_path):
    ## Load image.
    X_img = np.array(Image.open(image), dtype=np.float32)
    X_img = X_img / np.max(X_img)
    img_height, img_width = X_img.shape
    assert img_height >= DIM_H and img_width >= DIM_W, "Image is too small"
    assert img_height < 2 * DIM_H and img_width < 2 * DIM_W, "Image is too large"

    X_img = X_img.reshape(img_height, img_width, 1)
    print("Loaded image: %s" % (image))

    ## Fragment the image.
    print("Fragmenting image for prediction.")

    TL = X_img[:DIM_H, :DIM_W][None]
    TR = X_img[:DIM_H, -DIM_W:][None]
    BL = X_img[-DIM_H:, :DIM_W][None]
    BR = X_img[-DIM_H:, -DIM_W:][None]
    predictedTL = model.predict(TL).squeeze()
    predictedBR = model.predict(BR).squeeze()
    predictedBL = model.predict(BL).squeeze()
    predictedTR = model.predict(TR).squeeze()

    # Reconstruct predictions.
    print("Reconstructing predictions to full sized image.")
    reconstructed_prediction = np.zeros((img_height, img_width), dtype=np.float32)

    mid_w = int(np.ceil(img_width / 2.0))
    mid_h = int(np.ceil(img_height / 2.0))

    reconstructed_prediction[:mid_h, :mid_w] = predictedTL[:mid_h, :mid_w]
    reconstructed_prediction[:mid_h, -mid_w:] = predictedTR[:mid_h, -mid_w:]
    reconstructed_prediction[-mid_h:, :mid_w] = predictedBL[-mid_h:, :mid_w]
    reconstructed_prediction[-mid_h:, -mid_w:] = predictedBR[-mid_h:, -mid_w:]

    imageio.imwrite(
        f"{output_path}/outfile{os.path.basename(image)[:-4]}.jpg",
        img_as_ubyte(reconstructed_prediction),
    )


def main():
    args = get_args()
    model = build_model(DIM_H, DIM_W)

    # checks
    assert not (
        args.image and args.folder
    ), "Please provide either image or folder path"
    assert args.image or args.folder, "Please provide either image or folder path"
    if not args.output_path:
        args.output_path = args.folder
    if args.image:
        images = [args.image]
    else:
        images = glob.glob(args.folder + "/*tif")

    for image in images:
        predict_full_image(model, image, args.output_path)


if __name__ == "__main__":
    main()
