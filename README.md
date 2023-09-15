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
