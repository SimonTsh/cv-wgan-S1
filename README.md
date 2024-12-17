# Complex-valued Wasserstein GAN for Sentinel-1 Image Superresolution
Complex-WGAN network adapted from https://github.com/jeremyfix/complex-wgan.git and modified for Sentinel-1 superresolution purpose. This repository provides the inference code as well as pretrained networks for the complex-wgan paper. 

To test inferences, you can proceed by 

1- creating a virtual environment

```
cd Code
python3 -m venv venv
source venv/bin/activate
```

2- Install the required dependencies

```
python -m pip install -r requirements.txt
```

3- Execute the sampling and interpolation script

```
python test_sample.py --load_model WGAN_fft_MNIST_16 --transform inverse_fft_contrast_enhanced
python test_sample.py --load_model WGAN_fft_FashionMNIST_2 --transform inverse_fft_contrast_enhanced
python test_sample.py --load_model SAR_WGAN_28 
```

4- Computes the FIDs

```
python utils/fid.py  --load_model WGAN_fft_MNIST_16 --postprocess ifft --fold train
python utils/fid.py  --load_model WGAN_fft_MNIST_16 --postprocess ifft --fold test
python utils/fid.py  --load_model WGAN_fft_FashionMNIST_2 --postprocess ifft --fold train
python utils/fid.py  --load_model WGAN_fft_FashionMNIST_2 --postprocess ifft --fold test
python utils/fid.py  --load_model SAR_WGAN_28 --postprocess "None" --fold train
python utils/fid.py  --load_model SAR_WGAN_28 --postprocess "None" --fold test
```

# Example interpolations

## On MNIST and FashioMNIST, in the Fourier domain

**Left** Interpolation in the latent space given three independent latent vectors $z_0, z_1, z_2$. The generated samples associated with $z_0$, $z_1$ and $z_2$ are respectively in the top left, top right and bottom left corners. 

**Right** Interpolation in the latent space by applying a rotation of a random latent vector $z_0$. 

For all these representations, the Fourier representation is transformed with an inverse Fourier transform in the image domain for easier interpretation.

![alt text](https://github.com/jeremyfix/complex-wgan/blob/main/images/Fig3_left.png?raw=true)
![alt text](https://github.com/jeremyfix/complex-wgan/blob/main/images/Fig3_right.png?raw=true)

## On SAR

**Left** Interpolation in the latent space given three independent latent vectors $z_0, z_1, z_2$. The generated samples associated with $z_0$, $z_1$ and $z_2$ are respectively in the top left, top right and bottom left corners. 

**Right** Interpolation in the latent space by applying a rotation of a random latent vector $z_0$. 

For all these representations, the complex numbers are represented in the HSV colorspace.

![alt text](https://github.com/jeremyfix/complex-wgan/blob/main/images/Fig6_hsv_left_bis.png?raw=true)
![alt text](https://github.com/jeremyfix/complex-wgan/blob/main/images/Fig6_hsv_right.png?raw=true)
