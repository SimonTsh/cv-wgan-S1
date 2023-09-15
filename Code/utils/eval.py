import torch
import os
from torchvision.utils import make_grid, save_image
import math
from . import logs


class EvaluationGenerator:
    def __init__(self, generator, saving_path=None) -> None:
        self.generator = generator
        self.dtype = generator.dtype
        self.latent_dim = generator.latent_dim
        self.saving_path = saving_path

        if self.saving_path != None and not (os.path.exists(self.saving_path)):
            os.mkdir(self.saving_path)

        self.max_image = 100

    def _get_file_path(self, figure_name):
        """Return the file path of the figure such as
        file_name = figure_name_<i_min>.png
        """
        i_min = 0
        while i_min < self.max_image:
            file_name = f"{figure_name}_{i_min}.png"
            file_path = os.path.join(self.saving_path, file_name)
            if not (os.path.exists(file_path)):
                return file_path
            i_min += 1
        return f"{figure_name}.png"

    def _save_image(self, img, figure_name):
        """Save image if self.saving_path is not None."""
        if self.saving_path != None:
            file_path = self._get_file_path(figure_name)
            save_image(img, file_path)

    def interpolate(self, N: int = 8, transform=None):
        """Interpolate in latent space and generate NxN images.
        Apply transform to the output of the generator.
        """
        FIGURE_NAME = "interpolate"

        latent_edges = torch.randn(3, self.latent_dim, dtype=self.dtype)

        latent_vectors_1 = (latent_edges[2] - latent_edges[0]).tile(N, N, 1)
        latent_vectors_2 = (latent_edges[1] - latent_edges[0]).tile(N, N, 1)

        t = torch.linspace(0, 0.5, N)
        grid_x, grid_y = torch.meshgrid(t, t, indexing="xy")
        grid_x = grid_x.unsqueeze(-1)
        grid_y = grid_y.unsqueeze(-1)

        points = (
            latent_edges[0] + grid_x * latent_vectors_1 + grid_y * latent_vectors_2
        ).view(-1, self.latent_dim)

        with torch.no_grad():
            if transform is not None:
                generated_img = transform(self.generator.from_noise(points))
                img = make_grid(generated_img, nrow=N, scale_each=True)
                self._save_image(img, FIGURE_NAME)
            else:
                generated_img = self.generator.from_noise(points).cpu()
                generated_img = (
                    logs.hsv_colorscale(generated_img).squeeze()
                    if generated_img.dtype == torch.complex64
                    else generated_img
                )

                img = make_grid(generated_img, nrow=N, scale_each=True)
                self._save_image(img, FIGURE_NAME + "_raw")

        return img

    def interpolate_circle(self, N: int = 8, transform=None):
        """Interpolate in latent space and generate NxN images.
        Apply transform to the output of the generator.
        """
        FIGURE_NAME = "interpolate_circle"

        latent_edges = torch.randn(1, self.latent_dim, dtype=self.dtype).tile(N, 1)
        t = torch.linspace(0, 2 * torch.pi, N).unsqueeze(1)

        points = latent_edges * torch.exp(t * 1j)

        if transform is not None:
            generated_img = transform(self.generator.from_noise(points))
            img = make_grid(generated_img, nrow=int(math.sqrt(N)), scale_each=True)
            self._save_image(img, FIGURE_NAME)

        else:
            generated_img = self.generator.from_noise(points).cpu()
            generated_img = (
                logs.hsv_colorscale(generated_img).squeeze()
                if generated_img.dtype == torch.complex64
                else generated_img
            )

            img = make_grid(generated_img, nrow=int(math.sqrt(N)), scale_each=True)
            self._save_image(img, FIGURE_NAME + "_raw")

        return img

    def interpolate_module(self, N: int = 8, transform=None):
        """Interpolate in latent space and generate NxN images.
        Apply transform to the output of the generator.
        """
        FIGURE_NAME = "interpolate_module"

        latent_edges = torch.randn(1, self.latent_dim, dtype=self.dtype).tile(4 * N, 1)
        t_abs = torch.linspace(-1, 1, 4 * N).unsqueeze(1)
        t_phase = torch.zeros_like(t_abs)
        t_phase[t_abs < 0] = torch.pi

        points = latent_edges * torch.abs(t_abs) * torch.exp(t_phase * 1j)

        if transform is not None:
            generated_img = transform(self.generator.from_noise(points))
            img = make_grid(generated_img, nrow=N, scale_each=True)
            self._save_image(img, FIGURE_NAME)
        else:
            generated_img = self.generator.from_noise(points).cpu()
            generated_img = (
                logs.hsv_colorscale(generated_img).squeeze()
                if generated_img.dtype == torch.complex64
                else generated_img
            )

            img = make_grid(generated_img, nrow=N, scale_each=True)
            self._save_image(img, FIGURE_NAME + "_raw")

        return img

    def plot(self, N: int = 8, transform=None, save_points=False):
        """Plot NxN images sampled randomly from latent space.
        if save_points=True, save the random points in latent space.
        """
        FIGURE_NAME = "sampled"

        latent_points = torch.randn(N * N, self.latent_dim, dtype=self.dtype)
        if save_points:
            points_path = self.saving_path + "points_sampled"
            torch.save(latent_points, points_path)

        if transform is not None:
            generated_img = transform(self.generator.from_noise(latent_points))
            img = make_grid(generated_img, nrow=N, scale_each=True)
            self._save_image(img, FIGURE_NAME)
        else:
            generated_img = self.generator.from_noise(latent_points).cpu()
            generated_img = (
                logs.hsv_colorscale(generated_img).squeeze()
                if generated_img.dtype == torch.complex64
                else generated_img
            )

            img = make_grid(generated_img, nrow=N, scale_each=True)
            self._save_image(img, FIGURE_NAME + "_raw")

        return img, latent_points
