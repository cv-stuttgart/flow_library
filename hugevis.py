import os
import cv2 as cv
import numpy as np
from PIL import Image
import datasets


def visualize(input_data, key="images"):
    """
    create a visualization for a whole dataset
    input_data: either a dictionary retrieved from datasets.getTrainDataset or a folder path
    key: the key to retrive the files from the dictionary, e.g. "images" or "flows"
    """
    width = 1920
    left_boundary = 120
    image_downsampled_width = 50
    bg_color = (20, 20, 20)
    text_color = (255, 255, 255)

    ###

    if type(input_data) == str:
        dataset = {}
        for sq in sorted(os.listdir(input_data)):
            sq_path = os.path.join(input_data, sq)
            dataset[sq] = {key: sorted([os.path.join(sq_path, x) for x in os.listdir(sq_path)])}
    elif type(input_data) == dict:
        # nothing to do
        dataset = input_data
    else:
        raise ValueError("Unknown input data:", type(input_data))

    first_im = dataset[list(dataset.keys())[0]][key][0]

    first_im = np.array(Image.open(first_im))
    im_height, im_width, _ = first_im.shape
    im_aspectratio = im_width / float(im_height)

    draw_width = width - left_boundary
    images_per_row = draw_width // image_downsampled_width
    image_downsampled_height = int(image_downsampled_width / im_aspectratio)

    # compute total height
    total_height = 0
    for sq, content in dataset.items():
        #sq_path = os.path.join(folderpath, sq)

        images = content[key] #os.listdir(sq_path)
        num_rows = len(images) // images_per_row + 1
        total_height += num_rows * image_downsampled_height

    total_height = int(total_height)

    result = np.zeros((total_height, width, 3), dtype=np.uint8)
    result[:, :, 0] = bg_color[0]
    result[:, :, 1] = bg_color[1]
    result[:, :, 2] = bg_color[2]

    current_height = 0

    for i, (sq, content) in enumerate(dataset.items()):
        images = content[key]
        num_rows = len(images) // images_per_row + 1

        (_, h), _ = (cv.getTextSize(sq, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1))
        cv.putText(result, sq, (10, int(current_height + 0.5 * num_rows * image_downsampled_height + h / 2)), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv.LINE_AA)

        for j, im in enumerate(images):
            im = np.array(Image.open(im))
            res = cv.resize(im, dsize=(image_downsampled_width, image_downsampled_height), interpolation=cv.INTER_LINEAR)
            row = j // images_per_row
            col = j % images_per_row

            result[current_height + row * image_downsampled_height:current_height + image_downsampled_height + row * image_downsampled_height, left_boundary+col*50:left_boundary+col*50+50,:] = res

        current_height += num_rows * image_downsampled_height

    Image.fromarray(result).save("test.png")


if __name__ == "__main__":
    # visualize("/local/datasets/mpi_sintel/training/clean")
    sintel = datasets.getTrainDataset("mpi_sintel", sintel_imagetype="clean")
    visualize(sintel)