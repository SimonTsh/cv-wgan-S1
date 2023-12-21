# Complex-valued Wasserstein GAN for SAR Images generation

This repository provides the inference code as well as pretrained networks for our paper. 

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

**Left** Interpolation in the latent space given three independent latent vectors $z_0, z_1, z_2$. The generated samples associated with $z_0$, $z_1$ and $z_2$ are respectively in the top left, top right and bottom left corners. 

**Right** Interpolation in the latent space by applying a rotation of a random latent vector $z_0$. 

For all these representations, the complex numbers are represented in the HSV colorspace.

![alt text](https://github.com/jeremyfix/complex-wgan/blob/main/images/Fig3_left.png?raw=true)
![alt text](https://github.com/jeremyfix/complex-wgan/blob/main/images/Fig3_right.png?raw=true)
