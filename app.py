import streamlit as st
from stqdm import stqdm
import time
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import gradio as gr
import time
import sys
import subprocess
import shutil
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *
import fastai
from datetime import datetime

def GFPGAN_SYS(face_restore_check,scaleup_check,color_restore_check,Facemodel,face_size_value,face_weight_value):
    """Inference demo for GFPGAN (for users).
    """
    parser = argparse.ArgumentParser()
    # 배경 , 얼굴 , 색상 복원
    if face_restore_check == True and scaleup_check == True and color_restore_check == True:
        parser.add_argument(
            '-i',
            '--input',
            type=str,
            #default='GFPGAN_lib/inputs/whole_imgs',
            default='frames',
            help='Input image or folder. Default: frames')
        parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    # 배경 , 얼굴 복원
    if face_restore_check == True and scaleup_check == True and color_restore_check == False:
        parser.add_argument(
            '-i',
            '--input',
            type=str,
            #default='GFPGAN_lib/inputs/whole_imgs',
            default='source',
            help='Input image or folder. Default: frames')
        parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')
    # 얼굴 , 색상 복원
    if face_restore_check == True and scaleup_check == False and color_restore_check == True:
        parser.add_argument(
            '-i',
            '--input',
            type=str,
            #default='GFPGAN_lib/inputs/whole_imgs',
            default='source',
            help='Input image or folder. Default: source')
        parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')   
    # 얼굴 복원
    if face_restore_check == True and scaleup_check == False and color_restore_check == False:
        parser.add_argument(
            '-i',
            '--input',
            type=str,
            #default='GFPGAN_lib/inputs/whole_imgs',
            default='source',
            help='Input image or folder. Default: source')
        parser.add_argument('-o', '--output', type=str, default='results', help='Output folder. Default: results')

    
    # we use version to select models, which is more user-friendly
    if Facemodel == "GFPGANv1.3":
        parser.add_argument(
            '-v', '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    if Facemodel == "GFPGANCleanv1-NoCE-C2":
        parser.add_argument(
            '-v', '--version', type=str, default='1.2', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    if Facemodel == "GFPGANv1":
        parser.add_argument(
            '-v', '--version', type=str, default='1', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    if Facemodel == "GFPGANv1.4":
        parser.add_argument(
            '-v', '--version', type=str, default='1.4', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    if Facemodel == "RestoreFormer":
        parser.add_argument(
            '-v', '--version', type=str, default='RestoreFormer', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    
    parser.add_argument(
        '-s', '--upscale', type=int, default=int(face_size_value), help='The final upsampling scale of the image. Default: 2')

    parser.add_argument(
        '--bg_upsampler', type=str, default='realesrgan', help='background upsampler. Default: realesrgan')
    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=float(face_weight_value), help='Adjustable weights.')          # 0.5  0.2
    args = parser.parse_args()
    #args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):
        args.input = args.input[:-1]
    if os.path.isfile(args.input):
        img_list = [args.input]
    else:
        img_list = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.output, exist_ok=True)
    # ------------------------ set up background upsampler ------------------------
    bg_upsampler = None
    # ------------------------ set up GFPGAN restorer ------------------------
    if args.version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif args.version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif args.version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif args.version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif args.version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {args.version}.')

    # determine model paths
    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        # download pre-trained models from url
        model_path = url

    restorer = GFPGANer(
        model_path=model_path,
        upscale=args.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # ------------------------ restore ------------------------
    
    for img_path in stqdm(img_list):
        # read image
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # restore faces and background if necessary
        # args.weight   # 얼굴 가중치
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=args.aligned,
            only_center_face=args.only_center_face,
            paste_back=True,
            weight=args.weight)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            # save cropped face
            save_crop_path = os.path.join(args.output, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            # save restored face
            if args.suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(args.output, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(args.output, 'cmp', f'{basename}_{idx:02d}.png'))

        # save restored img
        if restored_img is not None:
            if args.ext == 'auto':
                extension = ext[1:]
            else:
                extension = args.ext

            if args.suffix is not None:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}_{args.suffix}.{extension}')
            else:
                save_restore_path = os.path.join(args.output, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)
    print(f'Results are in the [{args.output}] folder.')
    #process_markdown.empty()

def FACE_IMAGE_SCALEUP_SYS(face_restore_check,scaleup_check,color_restore_check,face_enhance_select,resize_value,Esrmodel,denoise_value,Facemodel):
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    
    # 배경 복원
    if face_restore_check == False and scaleup_check == True and color_restore_check == False:
        parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    # 배경 복원 , 컬러 복원
    if face_restore_check == False and scaleup_check == True and color_restore_check == True:
        parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    # 배경 복원 , 얼굴 복원
    if face_restore_check == True and scaleup_check == True and color_restore_check == False:
        parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    # 배경 복원 , 얼굴 복원 , 색상 복원
    if face_restore_check == True and scaleup_check == True and color_restore_check == True:
        parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')

    if Esrmodel == "RealESRGAN_x4plus":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRGAN_x4plus',
            help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
                'realesr-animevideov3 | realesr-general-x4v3'))
    
    if Esrmodel == "RealESRGAN_x2plus":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRGAN_x2plus',
            help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
                'realesr-animevideov3 | realesr-general-x4v3'))
    
    if Esrmodel == "realesr-general-x4v3":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='realesr-general-x4v3',
            help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
                'realesr-animevideov3 | realesr-general-x4v3'))
    
    if Esrmodel == "RealESRNet_x4plus":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRNet_x4plus',
            help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
                'realesr-animevideov3 | realesr-general-x4v3'))
    
    if Esrmodel == "RealESRGAN_x4plus_anime_6B":
        parser.add_argument(
            '-n',
            '--model_name',
            type=str,
            default='RealESRGAN_x4plus_anime_6B',
            help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
                'realesr-animevideov3 | realesr-general-x4v3'))
    
    
    parser.add_argument('-o', '--output', type=str, default='scaleup', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=float(denoise_value),
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=int(resize_value), help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.fp32,
        gpu_id=args.gpu_id)

    
    #if args.face_enhance:  # Use GFPGAN for face enhancement
    from gfpgan import GFPGANer
    
    if Facemodel == "GFPGANv1.3":
        face_enhancer = GFPGANer(
            model_path=r'C:\Users\emine\AppData\Local\Programs\Python\Python311\Lib\site-packages\gfpgan\weights\GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    
    if Facemodel == "GFPGANCleanv1-NoCE-C2":
        face_enhancer = GFPGANer(
            model_path=r'C:\Users\emine\AppData\Local\Programs\Python\Python311\Lib\site-packages\gfpgan\weights\GFPGANCleanv1-NoCE-C2.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    
    #if Facemodel == "GFPGANv1":
        #face_enhancer = GFPGANer(
            #model_path=r'C:\Users\emine\AppData\Local\Programs\Python\Python311\Lib\site-packages\gfpgan\weights\GFPGANv1.pth',
            #upscale=args.outscale,
            #arch='clean',
            #channel_multiplier=2,
            #bg_upsampler=upsampler)
    
    if Facemodel == "GFPGANv1.4":
        face_enhancer = GFPGANer(
            model_path=r'C:\Users\emine\AppData\Local\Programs\Python\Python311\Lib\site-packages\gfpgan\weights\GFPGANv1.4.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)
    
    #if Facemodel == "RestoreFormer":
        #face_enhancer = GFPGANer(
            #model_path=r'C:\Users\emine\AppData\Local\Programs\Python\Python311\Lib\site-packages\gfpgan\weights\RestoreFormer.pth',
            #upscale=args.outscale,
            #arch='clean',
            #channel_multiplier=2,
            #bg_upsampler=upsampler)

    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(stqdm(paths)):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        if face_enhance_select == True:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        if face_enhance_select == False:
            output, _ = upsampler.enhance(img, outscale=args.outscale)
        
        if args.ext == 'auto':
            extension = extension[1:]
        else:
            extension = args.ext
        if img_mode == 'RGBA':  # RGBA images should be saved in png format
            extension = 'png'
        if args.suffix == '':
            save_path = os.path.join(args.output, f'{imgname}.{extension}')
        else:
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
        cv2.imwrite(save_path, output)

    
def DEOLDIFY_SYS(color_value):
    device.set(device=DeviceId.GPU0)
    #if not torch.cuda.is_available():
    #print('GPU not available.')
    torch.cuda.is_available()

    colorizer = get_image_colorizer(artistic=False)
    source_url = None #@param {type:"string"}
    render_factor = int(color_value)  #@param {type: "slider", min: 7, max: 40}
    watermarked = True #@param {type:"boolean"}
    img_root= r'C:\Users\emine\Documents\work\20231215_work\test_images'
    
    i = 0
    for img_path in stqdm(os.listdir(img_root)):
        source_path = os.path.join(img_root, img_path)
        i+=1
        print(i,source_path)
        result_path = None

        if source_url is not None:
            colorizer.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=True, watermarked=watermarked)
        else:
            colorizer.plot_transformed_image(source_path, render_factor=render_factor, display_render_factor=True, figsize=(8,8), compare=True)
    
def main():
    st.set_page_config(layout="wide")
    st.markdown("""<style>.big-font {font-size:50px !important;}</style>""", unsafe_allow_html=True)
    st.markdown('<p class="big-font">LastHouse Restoration</p>', unsafe_allow_html=True)
    Image_input = st.text_input('Input Directory','')
    Image_output = st.text_input('Output Directory','')
    Upscale_check = st.checkbox('UpScale')
    Resize_value = st.slider("Resize", 0,8,1)
    Real_ESRModel = st.selectbox(
        'Model',
        ('RealESRGAN_x4plus','RealESRGAN_x2plus','realesr-general-x4v3','RealESRNet_x4plus','RealESRGAN_x4plus_anime_6B')
    )
    Denoise_value = st.slider("Denoising strength (only use realesr-general-x4v3)", 0.0,1.0,0.1)
    Face_Restoration_check = st.checkbox('Face Restoration')
    Face_Model = st.selectbox(
        'Face Model',
        ('GFPGANv1.3','GFPGANCleanv1-NoCE-C2','GFPGANv1.4')
    )
    Face_restore_size = st.slider("Face restore size(only use UpScale No check)", 0,8,1)
    Face_restore_weight = st.slider("Face restore size(only use UpScale No check)", 0.0,5.0,0.1)
    Color_check = st.checkbox('Color')
    Color_value = st.slider("Coloring", 0,40,1)
    
    copycmd = "copy"
    input_file_line = Image_input+"\*"
    input_file_line2 = r"C:\Users\emine\Documents\work\20231215_work\source"
    input_file_cmd = copycmd+" "+input_file_line+" "+input_file_line2

    markdown_vi = st.empty()
    markdown_vi.write("")

    # 얼굴 복원 파일 경로
    if Upscale_check == False and Face_Restoration_check == True and Color_check == False:
        Face_restore_file = r"C:\Users\emine\Documents\work\20231215_work\results\restored_imgs"
        Face_restore_file1 = Face_restore_file+"\*"
        Face_restore_file2 = Image_output
        Face_restore_file_cmd = copycmd+" "+Face_restore_file1+" "+Face_restore_file2
    
    # 배경 복원 파일 경로
    if Upscale_check == True and Face_Restoration_check == False and Color_check == False:
        Upscale_image_file = r"C:\Users\emine\Documents\work\20231215_work\source"
        Upscale_image_file1 = Upscale_image_file+"\*"
        Upscale_image_file2 = r"C:\Users\emine\Documents\work\20231215_work\inputs"
        Upscale_image_file_cmd = copycmd+" "+Upscale_image_file1+" "+Upscale_image_file2

        Upscale_image_file3 = r"C:\Users\emine\Documents\work\20231215_work\scaleup"
        Upscale_image_file4 = Upscale_image_file3+"\*"
        Upscale_image_file5 = Image_output
        Upscale_image_file_cmd1 = copycmd+" "+Upscale_image_file4+" "+Upscale_image_file5
    
    # 컬러 복원 경로
    if Upscale_check == False and Face_Restoration_check == False and Color_check == True:
        Image_color_restore_file = r"C:\Users\emine\Documents\work\20231215_work\source"
        Image_color_restore_file1 = Image_color_restore_file+"\*"
        Image_color_restore_file2 = r"C:\Users\emine\Documents\work\20231215_work\test_images"
        Image_color_restore_file_cmd = copycmd+" "+Image_color_restore_file1+" "+Image_color_restore_file2

        Image_color_restore_file3 = r"C:\Users\emine\Documents\work\20231215_work\result_images"
        Image_color_restore_file4 = Image_color_restore_file3+"\*"
        Image_color_restore_file5 = Image_output
        Image_color_restore_file_cmd1 = copycmd+" "+Image_color_restore_file4+" "+Image_color_restore_file5

    # 배경 복원 , 컬러 복원 경로
    if Upscale_check == True and Face_Restoration_check == False and Color_check == True:
        Upscale_file = r"C:\Users\emine\Documents\work\20231215_work\source"
        Upscale_file1 = Upscale_file+"\*"
        Upscale_file2 = r"C:\Users\emine\Documents\work\20231215_work\inputs"
        Upscale_file_cmd = copycmd+" "+Upscale_file1+" "+Upscale_file2
        
        Upscale_file3 = r"C:\Users\emine\Documents\work\20231215_work\scaleup"
        Upscale_file4 = Upscale_file3+"\*"
        Upscale_file5 = r"C:\Users\emine\Documents\work\20231215_work\test_images"
        Upscale_file_cmd1 = copycmd+" "+Upscale_file4+" "+Upscale_file5
        
        Upscale_file6 = r"C:\Users\emine\Documents\work\20231215_work\result_images"
        Upscale_file7 = Upscale_file6+"\*"
        Upscale_file8 = Image_output
        Upscale_file_cmd2 = copycmd+" "+Upscale_file7+" "+Upscale_file8
    
    # 얼굴, 색상 복원
    if Upscale_check == False and Face_Restoration_check == True and Color_check == True:
        Color_restore_file = r"C:\Users\emine\Documents\work\20231215_work\results\restored_imgs"
        Color_restore_file1 = Color_restore_file+"\*"
        Color_restore_file2 = r"C:\Users\emine\Documents\work\20231215_work\test_images"
        Color_restore_file_cmd = copycmd+" "+Color_restore_file1+" "+Color_restore_file2

        Color_restore_file3 = r"C:\Users\emine\Documents\work\20231215_work\result_images"
        Color_restore_file4 = Color_restore_file3+"\*"
        Color_restore_file5 = Image_output
        Color_restore_file_cmd1 = copycmd+" "+Color_restore_file4+" "+Color_restore_file5
    
    # 얼굴 , 배경 복원
    if Upscale_check == True and Face_Restoration_check == True and Color_check == False:
        Image_scaleup_file = r"C:\Users\emine\Documents\work\20231215_work\source"
        Image_scaleup_file1 = Image_scaleup_file+"\*"
        Image_scaleup_file2 = r"C:\Users\emine\Documents\work\20231215_work\inputs"
        Image_scaleup_file_cmd = copycmd+" "+Image_scaleup_file1+" "+Image_scaleup_file2

        Image_scaleup_file3 = r"C:\Users\emine\Documents\work\20231215_work\scaleup"
        Image_scaleup_file4 = Image_scaleup_file3+"\*"
        Image_scaleup_file5 = Image_output
        Image_scaleup_file_cmd1 = copycmd+" "+Image_scaleup_file4+" "+Image_scaleup_file5
    
    #얼굴 , 배경 , 색상 복원
    if Upscale_check == True and Face_Restoration_check == True and Color_check == True:
        total_output_file_line = r"C:\Users\emine\Documents\work\20231215_work\source"
        total_output_file_line1 = total_output_file_line+"\*"
        total_output_file_line2 = r"C:\Users\emine\Documents\work\20231215_work\inputs"
        total_output_file_cmd = copycmd+" "+total_output_file_line1+" "+total_output_file_line2

        total_output_file_line3 = r"C:\Users\emine\Documents\work\20231215_work\scaleup"
        total_output_file_line4 = total_output_file_line3+"\*"
        total_output_file_line5 = r"C:\Users\emine\Documents\work\20231215_work\test_images"
        total_output_file_cmd1 = copycmd+" "+total_output_file_line4+" "+total_output_file_line5

        total_output_file_line6 = r"C:\Users\emine\Documents\work\20231215_work\result_images"
        total_output_file_line7 = total_output_file_line6+"\*"
        total_output_file_line8 = Image_output
        total_output_file_cmd2 = copycmd+" "+total_output_file_line7+" "+total_output_file_line8

    if st.button('Generate'):
        os.system(input_file_cmd)
        # 얼굴 복원
        if Upscale_check == False and Face_Restoration_check == True and Color_check == False:
            markdown_vi.write("얼굴 복원중..")
            GFPGAN_SYS(Face_Restoration_check,Upscale_check,Color_check,Face_Model,Face_restore_size,Face_restore_weight)
            os.system(Face_restore_file_cmd)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")
        
        # 이미지 복원
        if Upscale_check == True and Face_Restoration_check == False and Color_check == False:
            markdown_vi.write("배경 복원중..")
            os.system(Upscale_image_file_cmd)
            FACE_IMAGE_SCALEUP_SYS(Face_Restoration_check,Upscale_check,Color_check,False,Resize_value,Real_ESRModel,Denoise_value,Face_Model)
            os.system(Upscale_image_file_cmd1)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")
        
        # 컬러 복원
        if Upscale_check == False and Face_Restoration_check == False and Color_check == True:
            os.system(Image_color_restore_file_cmd)
            markdown_vi.write("색상 복원중..")
            DEOLDIFY_SYS(Color_value)
            os.system(Image_color_restore_file_cmd1)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")
        
        # 배경 복원, 컬러 복원 
        if Upscale_check == True and Face_Restoration_check == False and Color_check == True:
            os.system(Upscale_file_cmd)
            markdown_vi.write("배경 복원중..")
            FACE_IMAGE_SCALEUP_SYS(Face_Restoration_check,Upscale_check,Color_check,False,Resize_value,Real_ESRModel,Denoise_value,Face_Model)
            os.system(Upscale_file_cmd1)
            markdown_vi.write("색상 복원중..")
            DEOLDIFY_SYS(Color_value)
            os.system(Upscale_file_cmd2)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")
        
        # 얼굴 , 색상 복원
        if Upscale_check == False and Face_Restoration_check == True and Color_check == True:
            markdown_vi.write("얼굴 복원중..")
            GFPGAN_SYS(Face_Restoration_check,Upscale_check,Color_check,Face_Model,Face_restore_size,Face_restore_weight)
            os.system(Color_restore_file_cmd)
            markdown_vi.write("색상 복원중..")
            DEOLDIFY_SYS(Color_value)
            os.system(Color_restore_file_cmd1)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")
        
        # 얼굴 , 배경 복원
        if Upscale_check == True and Face_Restoration_check == True and Color_check == False:
            os.system(Image_scaleup_file_cmd)
            markdown_vi.write("얼굴,배경 복원중..")
            FACE_IMAGE_SCALEUP_SYS(Face_Restoration_check,Upscale_check,Color_check,True,Resize_value,Real_ESRModel,Denoise_value,Face_Model) 
            os.system(Image_scaleup_file_cmd1)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")
        
        #얼굴 , 배경 , 색상 복원
        if Upscale_check == True and Face_Restoration_check == True and Color_check == True:
            os.system(total_output_file_cmd)
            markdown_vi.write("얼굴,배경 복원중..")
            FACE_IMAGE_SCALEUP_SYS(Face_Restoration_check,Upscale_check,Color_check,True,Resize_value,Real_ESRModel,Denoise_value,Face_Model)
            os.system(total_output_file_cmd1)
            markdown_vi.write("색상 복원중..")
            DEOLDIFY_SYS(Color_value)
            os.system(total_output_file_cmd2)
            markdown_vi.write("복원 완료 되었습니다!")
            time.sleep(2)
            markdown_vi.write("")

if __name__ == "__main__":
    main()