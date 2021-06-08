
import os
import argparse

import torch
import torchvision.datasets as dset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from torch.utils.data import sampler


def load_data(data_dir: str = "./tmp", num_imgs: int = 25, random: bool = False) -> tuple:

    """
    Loads the cifar10 data.

    Args:
        data_dir:
            The directory to store the full CIFAR10 dataset.
        num_imgs:
            The number of images to extract from the test-set
        random:
            If true, random image indices are used, otherwise the first images
            are used.
    Returns:
        A tuple of tensors (images, labels).
    """

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    trns_norm = trans.ToTensor()
    cifar10_test = dset.CIFAR10(data_dir, train=False, download=True, transform=trns_norm)

    if random:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs,
                                 sampler=sampler.SubsetRandomSampler(range(10000)))
    else:
        loader_test = DataLoader(cifar10_test, batch_size=num_imgs)

    return next(iter(loader_test))


# noinspection PyShadowingNames
def create_input_bounds(img: torch.Tensor, eps: float,
                        mean: tuple = (0.4914, 0.4822, 0.4465),
                        std: tuple = (0.2471, 0.2435, 0.2616)) -> torch.Tensor:

    """
    Creates input bounds for the given image and epsilon.

    The lower bounds are calculated as img-eps clipped to [0, 1] and the upper bounds
    as img+eps clipped to [0, 1].

    Args:
        img:
            The image.
        eps:
           The maximum accepted epsilon perturbation of each pixel.
        mean:
            The channel-wise means.
        std:
            The channel-wise standard deviation.
    Returns:
        A  img.shape x 2 tensor with the lower bounds in [..., 0] and upper bounds
        in [..., 1].
    """

    mean = torch.Tensor(mean).view(-1, 1, 1)
    std = torch.Tensor(std).view(-1, 1, 1)

    bounds = torch.zeros((*img.shape, 2), dtype=torch.float32)
    bounds[..., 0] = (torch.clip((img - eps), 0, 1) - mean) / std
    bounds[..., 1] = (torch.clip((img + eps), 0, 1) - mean) / std
    # print(bounds[..., 0].abs().sum(), bounds[..., 1].abs().sum())

    return bounds.view(-1, 2)


# noinspection PyShadowingNames
def save_vnnlib(input_bounds: torch.Tensor, label: int, spec_path: str, total_output_class: int = 10):

    """
    Saves the classification property derived as vnn_lib format.

    Args:
        input_bounds:
            A Nx2 tensor with lower bounds in the first column and upper bounds
            in the second.
        label:
            The correct classification class.
        spec_path:
            The path used for saving the vnn-lib file.
        total_output_class:
            The total number of classification classes.
    """

    with open(spec_path, "w") as f:

        f.write(f"; CIFAR10 property with label: {label}.\n")

        # Declare input variables.
        f.write("\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(declare-const X_{i} Real)\n")
        f.write("\n")

        # Declare output variables.
        f.write("\n")
        for i in range(total_output_class):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n")

        # Define input constraints.
        f.write(f"; Input constraints:\n")
        for i in range(input_bounds.shape[0]):
            f.write(f"(assert (<= X_{i} {input_bounds[i, 1]}))\n")
            f.write(f"(assert (>= X_{i} {input_bounds[i, 0]}))\n")
            f.write("\n")
        f.write("\n")

        # Define output constraints.
        f.write(f"; Output constraints:\n")
        # orignal separate version:
        # for i in range(total_output_class):
        #     if i != label:
        #         f.write(f"(assert (>= Y_{label} Y_{i}))\n")
        # f.write("\n")

        # disjunction version:
        f.write("(assert (or\n")
        for i in range(total_output_class):
            if i != label:
                f.write(f"    (and (>= Y_{i} Y_{label}))\n")
        f.write("))\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_images', type=int, default=25)
    parser.add_argument('--random', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epsilons', type=str, default="8/255")
    args = parser.parse_args()

    try:
        num_imgs = args.num_images
        random = args.random
        print(args.random)
        epsilons = [eval(eps) for eps in args.epsilons.split(" ")]
    except ValueError:
        msg = "Error, usage: $python generate_properties --num_images <int> --random <bool> --epsilons <str> \n"
        msg += "Example: $python generate_properties --num_images 25 --random True --epsilons '0.03 0.05'"
        raise ValueError(msg)

    result_dir = "./vnnlib_properties"

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    if args.seed is not None:
        # we use random seed 0 for deterministic testing
        torch.random.manual_seed(args.seed)

    images, labels = load_data(num_imgs=num_imgs, random=random)

    for eps in epsilons:
        for i in range(num_imgs):

            image, label = images[i], labels[i]
            input_bounds = create_input_bounds(image, eps)

            spec_path = f"vnnlib_properties/prop_{i}_eps_{eps:.3f}.vnnlib"

            save_vnnlib(input_bounds, label, spec_path)