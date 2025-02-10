import os
import torch
import numpy as np
import wandb
import shutil
from torchvision.utils import make_grid
from skimage import exposure

N_IMAGES = 64


class GANCheckpoint:
    def __init__(
        self,
        generator,
        discriminator,
        logdir_path: str = "./logs/",
        run_name: str = "GAN",
        min_epoch: int = 0,
        step: int = 0,
    ) -> None:
        self.min_value = None
        self.logdir_path = logdir_path
        self.generator = generator
        self.discriminator = discriminator
        self.min_epoch = min_epoch  # Save models if epoch > min_epoch
        self.run_name = ""

        self.file_path = self.generate_unique_logpath(logdir_path, run_name)

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

        self.step = step
        self.count = 0  # Counter last model checkpoint

    def save_config(self, config_file_path):
        dest_config_file_path = os.path.join(self.file_path, "config.yaml")
        if not (os.path.exists(dest_config_file_path)):
            shutil.copyfile(config_file_path, dest_config_file_path)

    def get_file_path(self):
        return self.file_path

    def generate_unique_logpath(self, dir, raw_run_name):
        if not os.path.exists(dir):
            os.mkdir(dir)

        log_path = os.path.join(dir, raw_run_name)
        if os.path.isdir(log_path):
            self.run_name = raw_run_name
            return log_path

        i = 0
        while True:
            run_name = raw_run_name + "_" + str(i)
            log_path = os.path.join(dir, run_name)
            if not os.path.isdir(log_path):
                self.run_name = run_name
                return log_path
            i = i + 1

    def save_models(self):
        # Save the model tensors parameters
        generator_path = os.path.join(self.file_path, f"{self.generator.name}.pt")
        torch.save(self.generator.state_dict(), generator_path)

        discriminator_path = os.path.join(
            self.file_path, f"{self.discriminator.name}.pt"
        )
        torch.save(self.discriminator.state_dict(), discriminator_path)

        # # Save the models as onnx
        # generator_path = os.path.join(self.file_path, f"{self.generator.name}.onnx")

        # # Export as onnx as well
        # # For the generator , the input size is (B, latent_dim)
        # #                     the output size is (B, 1, H, W)
        # device = next(self.generator.parameters()).device
        # dummy_latent = torch.zeros(
        #     (1, self.generator.latent_dim), device=device, dtype=self.generator.dtype
        # )
        # self.generator.eval()
        # torch.onnx.export(
        #     self.generator,
        #     dummy_latent,
        #     generator_path,
        #     verbose=False,
        #     opset_version=12,
        #     input_names=["input"],
        #     output_names=["output"],
        #     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        # )
        # # For the discriminator , the input size is (B, 1, H, W)
        # #                         the output size is (B, 1)
        # discriminator_path = os.path.join(
        #     self.file_path, f"{self.discriminator.name}.onnx"
        # )
        # device = next(self.generator.parameters()).device
        # dummy_input = torch.zeros(
        #     (
        #         1,
        #         self.discriminator.input_channels,
        #         self.discriminator.image_size,
        #         self.discriminator.image_size,
        #     ),
        #     device=device,
        #     dtype=self.discriminator.dtype,
        # )
        # self.discriminator.eval()
        # torch.onnx.export(
        #     self.discriminator,
        #     dummy_input,
        #     discriminator_path,
        #     verbose=False,
        #     opset_version=12,
        #     input_names=["input"],
        #     output_names=["output"],
        #     dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        # )

    def update_step(self):
        """
        Save models every <self.step> epochs.
        """
        if self.step > 0:
            self.count += 1
            if self.count % self.step == 0:
                self.save_models()
        else:
            print("Warning, checkpoint step <= 0")

    def update(self, loss, epoch):
        """
        Save models when <loss> is < to <self.min_value>.
        """
        if epoch >= self.min_epoch:
            if (self.min_value is None) or (loss < self.min_value):
                self.save_models()
                self.min_value = loss


class WandBLogger:
    """
    Wrapper for wandb logging.
    """

    def __init__(self, wandb_logger, name, transform_image=None, n_images=N_IMAGES):
        self.wandb_logger = wandb_logger
        self.name = name

        if transform_image == None or transform_image == "":
            self.transform_image = None
        else:
            self.transform_image = transform_image

        self.n_images = n_images
        self.seed = None

    def update_seed(self, latent_dim):
        """
        Update seed
        """
        if self.seed == None:
            if self.name == "sar":
                self.seed = torch.randn(
                    self.n_images, latent_dim, dtype=torch.complex64
                )
            else:
                self.seed = torch.randn(
                    self.n_images, latent_dim, dtype=torch.complex64
                )

    def log(self, dict, epoch):
        """
        Adds all scalars stored in a dictionary to the TensorBoard SummaryWriter.
        """
        self.wandb_logger.log(dict, epoch)

    def min_max_scale(self, tensor):
        """
        Apply minmax scaling to a tensor
        """
        return tensor - tensor.min() / (tensor.max() - tensor.min())

    def grid_real_images(self, loader):
        """
        Returns a grid of real images.
        """
        real_img = next(iter(loader))
        real_img = real_img[: self.n_images, ...]
        nrow = int(np.sqrt(self.n_images))
        grid = make_grid(real_img, nrow=nrow, padding=2, scale_each=True)

        return grid

    def grid_generated_images(self, generator, use_transform=True):
        """
        Returns a grid of generated images.
        """
        self.update_seed(generator.latent_dim)

        generator.eval()
        with torch.no_grad():
            img = generator.from_noise(self.seed).squeeze(0)

        if self.transform_image != None and use_transform:
            img = self.transform_image(img)
        else:
            if img.dtype == torch.complex64:
                img = torch.abs(img)

        nrow = int(np.sqrt(self.n_images))
        grid = make_grid(img, nrow=nrow, padding=2, scale_each=True)

        return grid

    def grid_compare_data_generated(self, generator, dataloader, mode="mod"):
        """
        Compare images of the dataset and generated images.
        mode = "real" for real part.
        mode = "imag" for imag part.
        mode = "mod" for module.
        mode = "phase" for phase.
        """
        ncol = int(np.sqrt(self.n_images))

        generator.eval()
        with torch.no_grad():
            fake_img = generator.generate(ncol)

        device = fake_img.device

        real_img = torch.zeros_like(fake_img, dtype=torch.complex64)
        for i, id in enumerate(torch.randint(len(dataloader.dataset), (ncol,))):
            real_img[i, :, :, :] = dataloader.dataset[int(id)][0]
        real_img = real_img.to(device)

        if mode == "real":
            stack = torch.cat(
                (
                    real_img.real,
                    fake_img.real,
                ),
                axis=0,
            )
        if mode == "imag":
            stack = torch.cat(
                (
                    real_img.imag,
                    fake_img.imag,
                ),
                axis=0,
            )
        if mode == "mod":
            stack = torch.cat(
                (
                    real_img.abs(),
                    fake_img.abs(),
                ),
                axis=0,
            )
        if mode == "phase":
            stack = torch.cat((real_img.angle(), fake_img.angle()), axis=0)

        grid = make_grid(stack, nrow=ncol, padding=2, scale_each=True)

        return grid

    def grid_compare_hsv_images(self, generator, dataloader):
        """Compare images in HSV color map.
        H for the phase and V for the module
        """
        ncol = int(np.sqrt(self.n_images))

        generator.eval()
        with torch.no_grad():
            fake_img = generator.generate(ncol)

        device = fake_img.device

        real_img = torch.zeros_like(fake_img, dtype=torch.complex64)
        for i, id in enumerate(torch.randint(len(dataloader.dataset), (ncol,))):
            try:
                real_img[i, :, :, :] = dataloader.dataset[int(id)][
                    0
                ]  # if (data, label)
            except:
                real_img[i, :, :, :] = dataloader.dataset[int(id)]

        real_img = real_img.to(device)

        stack = torch.cat(
            (hsv_colorscale(real_img.squeeze()), hsv_colorscale(fake_img.squeeze())),
            axis=0,
        )

        grid = make_grid(stack, nrow=ncol, padding=2, scale_each=True)

        return grid

    def write_image(self, img, epoch, desc=""):
        """
        Write an generated image using the same seed
        """
        image_name = f"{desc} - {epoch}"
        images = wandb.Image(img, caption=image_name)

        return {desc: images}


def hsv_to_rgb_torch(hsv):
    """
    Convert HSV values to RGB.
    Parameters
    ----------
    hsv : (..., 3) array-like
        All values assumed to be in range [0, 1]
    Returns
    -------
    (..., 3) `~numpy.ndarray`
        Colors converted to RGB values in range [0, 1]
    """

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError(
            "Last dimension of input array must be 3; "
            "shape {shp} was found.".format(shp=hsv.shape)
        )

    in_shape = hsv.shape

    hsv = hsv.clone().detach()

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    r = torch.empty_like(h)
    g = torch.empty_like(h)
    b = torch.empty_like(h)

    i = (h * 6.0).int()
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = i % 6 == 0
    r[idx] = v[idx]
    g[idx] = t[idx]
    b[idx] = p[idx]

    idx = i == 1
    r[idx] = q[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = i == 2
    r[idx] = p[idx]
    g[idx] = v[idx]
    b[idx] = t[idx]

    idx = i == 3
    r[idx] = p[idx]
    g[idx] = q[idx]
    b[idx] = v[idx]

    idx = i == 4
    r[idx] = t[idx]
    g[idx] = p[idx]
    b[idx] = v[idx]

    idx = i == 5
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = q[idx]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = torch.stack([r, g, b], dim=-1)

    return rgb.reshape(in_shape)  # [B, H, W, C]


def hsv_colorscale(img):
    # Calculate magnitude and phase of the image
    # https://www.jedsoft.org/fun/complex/fztopng.html according to this
    if isinstance(img, torch.Tensor):
        np_img = img.numpy()
    else:
        np_img = img
    np_mag = np.log(np.abs(np_img) + 1)
    p2, p98 = np.percentile(np_mag, (2, 98))
    np_mag = exposure.rescale_intensity(np_mag, in_range=(p2, p98))

    mag = torch.from_numpy(np_mag)

    phase = img.angle()

    # Normalize magnitude to [0,1] range
    # mag_norm = (mag - torch.min(mag)) / (torch.max(mag) - torch.min(mag))

    # Convert phase to [0,1] range
    phase_norm = (phase + torch.pi) / (2 * torch.pi)

    # Convert to HSV format
    hsv_image = torch.stack([phase_norm, mag / 2 + 0.5, mag], dim=-1)

    rgb = hsv_to_rgb_torch(hsv_image)

    return rgb.swapaxes(1, -1).swapaxes(-1, -2)  # [B, C, H, W]


def find_run_path(load_model, toplogdir):
    """
    Load a model from run logs.
    """

    if load_model != "":
        model_id = [
            run_name for run_name in os.listdir(toplogdir) if load_model in run_name
        ]
        if os.path.exists(load_model):
            if not ".pt" in load_model:
                logs_path = load_model
            else:
                logs_path = os.path.dirname(load_model)
        elif os.path.exists(os.path.join(toplogdir, load_model)):
            logs_path = os.path.join(toplogdir, load_model)
        elif len(model_id) == 1:  # Find model by id
            logs_path = os.path.join(toplogdir, model_id[0])
        else:
            print(f"Can't load model at {load_model}")
            logs_path = None

    return logs_path


def find_config(run_path):
    print(f"Looking for configs in {run_path}")
    config_file = [fn for fn in os.listdir(run_path) if "config.yaml" in fn]

    if len(config_file) == 1:
        config_path = os.path.join(run_path, config_file[0])
    else:
        config_path = None

    return config_path
