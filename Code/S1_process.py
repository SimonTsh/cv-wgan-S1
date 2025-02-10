import os
import glob
from PIL import Image
from tifffile import imread
from zipfile import ZipFile

import pickle
import numpy as np
import cv2

from scipy import interpolate, optimize

import matplotlib.pyplot as plt
from utils.patch_process import gaussian, win_patch, filter_2d, grid_patch_extraction, display_img, downsample_sar

def fit_curve(x, img):
    # coeff = np.polyfit(x, img_cut_abs_log, 2) # quadratic
    # img_fit = np.poly1d(coeff)(x)
    p0 = [np.max(img) * 3/4, np.mean(img), np.std(img), 0.0] # initial guesses
    popt, pcov = optimize.curve_fit(gaussian, x, img, p0=p0, maxfev=2000)
    img_fit = gaussian(x, *popt)

    return img_fit

def get_3dB_res(x, IPF):
    peak_amplitude = np.max(IPF) # Find the peak amplitude
    three_db_amplitude = peak_amplitude - 3 #* 0.707 # Calculate the 3dB amplitude
    diff = np.abs(IPF - three_db_amplitude) # indices where the impulse response crosses the -3dB amplitude
    indices = np.sort(np.argsort(diff)[:2]) # ensure the indices are in ascending order

    width_3dB = x[indices[1]] - x[indices[0]] # Calculate the 3dB width

    return width_3dB

def downres_generation(img_patch, radius, method='rect'):
    img_patch_f = np.fft.fftshift(np.fft.fft2(img_patch))
    img_filter_f, mask_f = filter_2d(img_patch_f, radius, method)
    img_filter = np.fft.ifft2(np.fft.ifftshift(img_filter_f))

    return img_patch_f, img_filter_f, img_filter

def draw_rectangle(x, y):
    global img
    # Define the rectangle size
    rectangle_width, rectangle_height = 100, 100
    
    start_x = x - rectangle_width // 2
    start_y = y - rectangle_height // 2
    end_x = start_x + rectangle_width
    end_y = start_y + rectangle_height
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    cv2.imshow('Rectangle', img)


working_dir = 'Code/data'
npy_dir = 'Code/npy'
action = 1 # 0: unzip downloaded data, 1: display unzipped data
patch_making = 1; # 0: only analysis, 1: create pickle dataset

if action == 0:
    zipfile_id = glob.glob(f'{working_dir}/*.zip') # ['6cd73e73-a10f-4598-b07b-f6018f53ef04.zip']
    for zipfile in zipfile_id:
        with ZipFile(f'{working_dir}/{zipfile}', 'r') as zip_ref:
            zip_ref.extractall(working_dir)
        try:
            os.remove(f'{working_dir}/{zipfile}') # Delete the original ZIP file
        except OSError as e:
            print(f"Error deleting file: {e}")

elif action == 1:
    folder_ids = glob.glob(f'{working_dir}/*.SAFE') # multi image processing
    # folder_id = 'S1A_S3_SLC__1SDH_20241120T213604_20241120T213629_056643_06F2DA_965E.SAFE' #'S1A_S6_SLC__1SDV_20241125T214410_20241125T214439_056716_06F5C2_86D5.SAFE'
    for folder_id in folder_ids:
        # for item in os.listdir(f'{working_dir}/{folder_id}/measurement'): # single image
        for item in os.listdir(f'{folder_id}/measurement'): # multi image processing
            file_name = item.split('.')[0]
            print(f'Processing {file_name}...')
            # Open the TIFF file
            # tiff_image = Image.open(f'{working_dir}/{folder_id}/measurement/{item}')
            # tiff_image = imread(f'{working_dir}/{folder_id}/measurement/{item}') # single image
            tiff_image = imread(f'{folder_id}/measurement/{item}') # multi image processing

            if patch_making:
                patch_size = 256 # 64 # 
                overlap_px = 0 # 10 # 50
                downsample_fac = 4
                patches_HR = grid_patch_extraction(tiff_image, [patch_size]*2, overlap_px)
                #win_patch(patches_HR, patch_size, patch_size//2, patch_size//2)
                patches_HR_f, img_LR_f, patches_LR = downres_generation(patches_HR, patch_size / downsample_fac, 'rect')
                # patches_LR = downsample_sar(patches_HR, downsample_fac) # 64 -> 16
                
                patch_index = 1000; tiny_e = 1 # 1e-15
                fig, axes = plt.subplots(2,2, dpi=300)
                axes[0,0].imshow(10*np.log10(np.abs(patches_HR[patch_index])+tiny_e),cmap='gray')
                axes[0,1].imshow(np.abs(patches_HR_f[patch_index]),cmap='gray')
                axes[1,1].imshow(np.abs(img_LR_f[patch_index]),cmap='gray')
                axes[1,0].imshow(10*np.log10(np.abs(patches_LR[patch_index])+tiny_e),cmap='gray')
                plt.tight_layout()
                fig.savefig(f'{working_dir}/images/patch_{file_name}.png')
                
                with open(f'{working_dir}/{file_name}_{patch_size}.pickle', 'wb') as file:
                    pickle.dump([patches_HR, patches_LR], file)
                print(f'Patches from {file_name} saved as dataset successfully...')
                
                break
            
            # Save as PNG --> image storage
            ## S3
            # SM (urban): [20000:26000,500:7500]
            # SM (ship): [2000:6000,2500:7500]
            # CR1: [3000:5000,2500:4500]
            # CR2: [2500:3500,6000:7000], centre around 150 with offset_x = 13, offset_y = -51
            # IW (urban): [5000:10000,:4000]
            # CR1: 
            
            ## S6
            # GRD (ship): [1000:4000,1000:4000]
            # SLC (ship): [4500:5500,3500:4500]
            # CR1: centre around 150 with offset_x = 190, offset_y = 78
            tiff_image_crop = tiff_image[20000:26000,500:7000] #rng_crop, azi_crop
            np.save(f'{npy_dir}/train/{file_name}.npy', tiff_image_crop) # save for further processing

            # For display
            img = display_img(tiff_image_crop)
            Image.fromarray(img).save(f'{working_dir}/images/{file_name}.png')

            # Method 1: patch by predefined center pixel
            window_size = 150 # in pixels
            offset_x = 13 #190
            offset_y = -51 #78
            img_patch = win_patch(tiff_image_crop, window_size, tiff_image_crop.shape[0]//2+offset_x, tiff_image_crop.shape[1]//2+offset_y)
            fig, axes = plt.subplots(2,2, dpi=300)
            axes[0,0].imshow(10*np.log10(np.abs(img_patch)+tiny_e),cmap='gray')
            
            # extract IPF
            cut_x = window_size//2 # extract center x cut
            img_cut = img_patch[cut_x,:] # range slice
            
            # upsample by n times
            rng_res = 3.55 # in m; calculate from c/2*Bw
            x = np.linspace(0, len(img_cut)*rng_res, len(img_cut))
            x_up = np.linspace(0, len(img_cut)*rng_res, len(img_cut) * 4) # upsample by 4
            img_cut_up = interpolate.interp1d(x, img_cut, kind='quadratic')(x_up)
            # axes[1,1].plot(10*np.log10(abs(img_cut_up)+tiny_e))
            
            # # perform filtering in freq domain
            # filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9  # np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]]) / 2 # np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 8
            # pad_size = np.array(img_patch.shape) - np.array(filter.shape)
            # filter_pad = np.pad(filter, ((0, pad_size[0]), (0, pad_size[1])), mode='constant')
            # filter_pad_f = np.fft.fft2(filter_pad)
            tiny_e = 1 # 1e-15
            upsample_factor = 4
            radius = window_size / upsample_factor #0.6 #[-0.35, 0.35, -0.35, 0.35]

            img_patch_f, img_filter_f, img_filter = downres_generation(img_patch, radius, 'rect')
            axes[0,1].imshow(np.abs(img_patch_f),cmap='gray')
            axes[1,1].imshow(np.abs(img_filter_f),cmap='gray')
            axes[1,0].imshow(10*np.log10(np.abs(img_filter)+tiny_e),cmap='gray')
            # axes[0,2].imshow(np.abs(mask_f),cmap='gray')
            plt.tight_layout()
            fig.savefig(f'{working_dir}/images/patch_{file_name}.png')
            
            # sin curve fitting
            img_cut_abs = np.abs(img_cut_up) #img_cut
            img_cut_fit = fit_curve(x_up, img_cut_abs) #x
            # img_fit[img_fit < 1] = np.nan
            img_cut_fit_3dBwidth = get_3dB_res(x_up, 10*np.log10(img_cut_abs / np.max(img_cut_abs))) #img_cut_fit
            print(f"The 3dB resolution (bef) is: {img_cut_fit_3dBwidth}")
            
            # extract range cut
            img_cut_filter = img_filter[cut_x,:]
            img_cut_filter_up = interpolate.interp1d(x, img_cut_filter, kind='quadratic')(x_up)
            
            img_filter_abs = np.abs(img_cut_filter_up) #img_cut_filter
            img_filter_fit = fit_curve(x_up, img_filter_abs) #x
            # img_filter_fit[img_filter_fit < 1] = np.nan
            img_filter_fit_3dBwidth = get_3dB_res(x_up, 10*np.log10(img_filter_abs / np.max(img_filter_abs))) #img_filter_fit
            print(f"The 3dB resolution (aft) is: {img_filter_fit_3dBwidth}")

            # visualise
            fig, axes = plt.subplots(1,2, dpi=300)
            axes[0].plot(x_up,10*np.log10(img_cut_abs / np.max(img_cut_abs)),label='IPF')
            axes[0].title.set_text(f'original IPF \n(res = {img_cut_fit_3dBwidth:.3f}m)')
            axes[0].set_ylabel('dB')
            # axes[0].plot(x_up,10*np.log10(img_cut_fit / np.max(img_cut_fit)),label='gaussian fit')
            axes[1].plot(x_up,10*np.log10(img_filter_abs / np.max(img_filter_abs)),label='widen IPF')
            axes[1].title.set_text(f'widen IPF \n(res = {img_filter_fit_3dBwidth:.3f}m)')
            # axes[1].plot(x_up,10*np.log10(img_filter_fit / np.max(img_filter_fit)),label='gaussian IPF')
            # axes[0,1].plot(10*np.log10(np.abs(np.fft.fft(img_cut_up))))
            plt.tight_layout()
            # fig.legend()
            fig.savefig(f'{working_dir}/images/IPF_{file_name}.png')

            # # Method 2: patch by drawing rect around pixels of interest
            # cv2.namedWindow('Rectangle')
            # cv2.setMouseCallback('Rectangle', draw_rectangle)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            # cv2.destroyAllWindows()
