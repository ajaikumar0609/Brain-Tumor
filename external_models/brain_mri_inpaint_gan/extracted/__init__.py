### python native imports
import os
import random
import math
from pathlib import Path
import logging
import sys

model_root_path = str(Path(os.path.abspath(__file__)).parent)
sys.path.insert(0, model_root_path)
sys.path.insert(0, model_root_path + '/Norm_slice') 
sys.path.insert(0, model_root_path + '/temp_slice') 

root_parent_path = str(Path(model_root_path).parent)
sys.path.insert(0, root_parent_path) # parent of parent
sys.path.insert(0, root_parent_path + '/Norm_slice') # parent of parent
sys.path.insert(0, root_parent_path + '/temp_slice') # parent of parent

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

### pypi imports
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
#from IPython.display import display


### local imports
from losses import *
from dataloader import *
from util import *
from model import Inpaint_generator, Tumor_shape, Tumor_grade

# device as global variable foor simplicity.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_models(tumor_shape=None,tumor_grade=None,inp_gen=None, inp_gen_path='inp_gen.pth'):
    """ Loading and initialising model checkpoints for tumor shape mask, tumo grade mask, and inpainting model. """
    
    if tumor_shape is None: 
        tumor_shape = Tumor_shape().to(device)
        try:
            tumor_shape = tumor_shape.load_state_dict(torch.load('tumor_shape.pth', map_location=device))
        except:
            tumor_shape.load_state_dict(torch.load(model_root_path + '/' + 'tumor_shape.pth', map_location=device))
        tumor_shape.eval()
    
    if tumor_grade is None:
        tumor_grade = Tumor_grade().to(device)
        try:
            tumor_grade.load_state_dict(torch.load('tumor_grade.pth', map_location=device))
        except:
            tumor_grade.load_state_dict(torch.load(model_root_path + '/' + 'tumor_grade.pth', map_location=device))
        tumor_grade.eval()

    if inp_gen is None or inp_gen_path != 'inp_gen.pth':
        inp_gen = Inpaint_generator().to(device)
        try:
            inp_gen.load_state_dict(torch.load(inp_gen_path, map_location=device))
        except:
            inp_gen.load_state_dict(torch.load(model_root_path + '/' + inp_gen_path, map_location=device))
        inp_gen.eval()
    logging.debug('tumor_shape, tumor_grade, inp_gen models were initialised and weights loaded.')
    return tumor_shape, tumor_grade, inp_gen

# initialize models
tumor_shape, tumor_grade, inp_gen = init_models()

def init_image_list():
    """ Lodaing the images of different MRI modalities that are used for inpainting"""
    
    # creation of an image list from which images will be loaded.
    try:
        F_default_img_list = glob.glob('temp_slice/F/*.png')
        if F_default_img_list is None or len(F_default_img_list) == 0:
            raise Exception
    except:
        F_default_img_list = glob.glob(model_root_path + '/temp_slice/F/*.png')
    try:
        T1_default_img_list = glob.glob('temp_slice/T1/*.png')
        if T1_default_img_list is None or len(T1_default_img_list) == 0:
            raise Exception
    except:
        T1_default_img_list = glob.glob(model_root_path + '/temp_slice/T1/*.png')
    try:
        T1c_default_img_list = glob.glob('temp_slice/T1c/*.png')
        if T1c_default_img_list is None or len(T1c_default_img_list) == 0:
            raise Exception
    except:
        T1c_default_img_list = glob.glob(model_root_path + '/temp_slice/T1c/*.png')
    try:
        T2_default_img_list = glob.glob('temp_slice/T2/*.png')
        if T2_default_img_list is None or len(T2_default_img_list) == 0:
            raise Exception
    except:
        T2_default_img_list = glob.glob(model_root_path + '/temp_slice/T2/*.png')


    logging.debug("lengths of lists of images imported from disk: len(F_default_img_list), len(T1_default_img_list), len(T1c_default_img_list), len(T2_default_img_list):", len(F_default_img_list), len(T1_default_img_list), len(T1c_default_img_list), len(T2_default_img_list))
    return F_default_img_list, T1_default_img_list, T1c_default_img_list, T2_default_img_list


# init input lists with default input images
F_default_img_list, T1_default_img_list, T1c_default_img_list, T2_default_img_list = init_image_list()


def read_images_for_inpainting(     
    F_img_list,
    T1_img_list,
    T1c_img_list,
    T2_img_list,
    index=0):
    """ Select an image and modality using an index (e.g. 10). Load that/these image(s) from disk. Later we will inpaint into these images. """
    
    #if F_img_list is None or T1_img_list is None or T1c_img_list is None or T2_img_list is None:
    #        init_image_list()

    temp_F = cv2.imread(F_img_list[index], 0) 
    temp_T1 = cv2.imread(T1_img_list[index], 0)
    temp_T1c = cv2.imread(T1c_img_list[index], 0)
    temp_T2 = cv2.imread(T2_img_list[index], 0)
    logging.debug('shapes of images from F_img_list:', temp_F.shape, '\n', 'range of img:', temp_F.min(), '~', temp_F.max())
    #display(Image.fromarray(temp_F))
    #display(Image.fromarray(temp_T1))
    #display(Image.fromarray(temp_T1c))
    #display(Image.fromarray(temp_T2))
    return temp_F, temp_T1, temp_T1c, temp_T2


def get_inpaints(
    num_samples, 
    image_size:tuple=(256, 256), 
    num_inpaints_per_sample:int=1,
    randomize_input_image_order:bool = True,
    max_index:bool = None,
    F_img_list=None,
    T1_img_list=None,
    T1c_img_list=None,
    T2_img_list=None,
    add_variations_to_mask:bool = True,
    x_center = 130,
    y_center = 130, 
    radius_1 = 10, 
    radius_2 = 15,
    radius_3 = 30
    ):

    transforms_ = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    #masks = []
    #inpaints = []
    output_list = []
    temp_list_of_indices = []
    for index in range(num_samples):

        # Setting the real index, as only max_index number of input images (<num_samples) are available.  
        real_index = index
        while real_index > max_index and max_index is not None:
            real_index = real_index - (max_index+1)
        logging.debug(f"index: {index}, real_index: {real_index}, max_index: {max_index}")
        
        # if max_index is not None, there is no need to randomize as all input images will be used anyway (and likely multiple times).
        if randomize_input_image_order and max_index is None:
            # random integer between 0 and num_samples
            real_index = random.randint(0, num_samples)
            # sample without replacement. 
            while real_index in temp_list_of_indices:
                # repeat until we find a missing index
                real_index = random.randint(0, num_samples)
            temp_list_of_indices.append(real_index)

        # get the 4 corresponding modality images. 
        temp_F, temp_T1, temp_T1c, temp_T2 = read_images_for_inpainting(F_img_list=F_img_list, T1_img_list=T1_img_list, T1c_img_list=T1c_img_list, T2_img_list=T2_img_list, index=real_index)


        for i in range(num_inpaints_per_sample):
            with torch.no_grad():
                # Set level circle with coodinates and radii to generate mask
                if add_variations_to_mask:
                    # temp circle with slight variations between iterations with slight variaton for make_level_circle's radii and circle centers x_center and y_center
                    all_radii_common_variation = random.uniform(-4, 2)
                    temp_circle = make_level_circle(
                        x_center = 130 + random.uniform(-12.5, 12.5), 
                        y_center = 130 + random.uniform(-12.5, 12.5), 
                        radius_1 = 10 + all_radii_common_variation + random.uniform(-2, 2), # common variation + specific variation  
                        radius_2 = 15 + all_radii_common_variation + random.uniform(-2, 2), # common variation + specific variation  
                        radius_3 = 30 + all_radii_common_variation + random.uniform(-2, 2), # common variation + specific variation  
                        image_height=image_size[0], 
                        image_width=image_size[1])
                else:
                    temp_circle = make_level_circle(
                        x_center = 130,
                        y_center = 130,
                        radius_1 = 10, 
                        radius_2 = 15,
                        radius_3 = 30,
                        image_height=image_size[0], 
                        image_width=image_size[1])
                #display(Image.fromarray(temp_circle[0][0].numpy()*255).convert('RGB'))
                #temp_circle.shape

                # Preparing a binary circle
                binary_circle = unify(temp_circle).reshape(1,1,image_size[0], image_size[1])
                binary_circle_torch = binary_circle.to(device)

                # Getting some information from the temp_F image
                uni_B = unify(temp_F).reshape(1,1,image_size[0], image_size[1])
                uni_B_torch = uni_B.to(device)

                # Preparing a binary mask using the tumor_shape model 
                binary_mask = unify(tumor_shape(torch.cat([uni_B_torch, binary_circle_torch], 1)).detach().cpu().numpy())
                binary_mask_torch = binary_mask.to(device)
                generated_binary_mask=Image.fromarray(binary_mask[0][0].numpy()*255).convert('RGB')
                #masks.append(generated_mask)

                # Preparing a concentric circle-based grade mask using the tumor_grade model
                temp_circle_torch = temp_circle.float().to(device)
                grade_mask = quantize(tumor_grade(torch.cat([binary_mask_torch, temp_circle_torch], 1)).detach().cpu().numpy())
                grade_mask_torch = grade_mask.to(device)
                generated_grade_mask=Image.fromarray(grade_mask[0][0].numpy()*255).convert('RGB')

                # Transforming images of different MRI modalities
                temp_F_torch = transforms_(temp_F).reshape(1,1,image_size[0],image_size[1])
                temp_T1_torch = transforms_(temp_T1).reshape(1,1,image_size[0],image_size[1])
                temp_T1c_torch = transforms_(temp_T1c).reshape(1,1,image_size[0],image_size[1])
                temp_T2_torch = transforms_(temp_T2).reshape(1,1,image_size[0],image_size[1])
                
                # Concatenation of the images
                brain_torch = torch.cat((temp_F_torch, temp_T1_torch, temp_T1c_torch, temp_T2_torch), 1).to(device)
                brain_blank_torch = (brain_torch * (1-binary_mask_torch))
                
                # Tumor inpainting using inp_gen model to all concatenated modalities at once
                out_brain_torch = inp_gen(brain_blank_torch, grade_mask_torch)

                 # Separating modalities again
                F_, T1_, T1c_, T2_ = torch.split(out_brain_torch.detach().cpu(),split_size_or_sections=1, dim=1)
                
                #print('results without merging')
                #generated_inp=Image.fromarray(F_[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                #output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "F")
                #output_list.append(output_as_tuple)
                #generated_inp=Image.fromarray(T1_[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                #output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "T1")
                #output_list.append(output_as_tuple)
                #generated_inp=Image.fromarray(T1c_[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                #output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "T1c")
                #output_list.append(output_as_tuple)
                #generated_inp=Image.fromarray(T2_[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                #output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "T2")
                #output_list.append(output_as_tuple)


                # merge with normal brain (at non-tumor parts)
                #print('results with merging')
                out_brain_final = out_brain_torch * binary_mask_torch + brain_torch * (1-binary_mask_torch)
                F_final, T1_final, T1c_final, T2_final = torch.split(out_brain_final.detach().cpu(),split_size_or_sections=1, dim=1)
                
                generated_inp=Image.fromarray(F_final[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "F")
                output_list.append(output_as_tuple)
                generated_inp=Image.fromarray(T1_final[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "T1")
                output_list.append(output_as_tuple)
                generated_inp=Image.fromarray(T1c_final[0][0].numpy()*255).convert('RGB') 
                #inpaints.append(generated_inp)
                output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "T1c")
                output_list.append(output_as_tuple)
                generated_inp=Image.fromarray(T2_final[0][0].numpy()*255).convert('RGB')
                #inpaints.append(generated_inp)
                output_as_tuple= (generated_inp, generated_binary_mask, generated_grade_mask, "T2")
                output_list.append(output_as_tuple)

                # outputs are to be returned as list of tuples (image, mask, label) to comply with medigan standards (e.g. see medigan.utils.split_images_masks_and_labels function) 

    return output_list



def normalise(data, nmax=1.0, nmin=0.0):
    """Image normalization of pixel values"""

    return (data - data.min()) * (
        (nmax - nmin) / (data.max() - data.min() + 1e-8)
    ) + nmin


def save_generated_inpaints(output_list, output_path, num_inpaints_per_sample):
    """Saving of the generated output_list (inpaints, masks, and label) in output_dir"""

    logging.debug(f"output_filepath: {output_path}")
    try:

        Path(output_path).mkdir(parents=True, exist_ok=True)

        for i, inpaint_tuple in enumerate(output_list):

            # sample num is the number of the input image used to generate the inpaint.
            # Note: int() this will cut off decimals,rounding towards lower digit (see https://stackoverflow.com/a/43661374)
            input_num = int(i/(4*num_inpaints_per_sample)) # 4 i per input sample. 
            inpaint_num = int(i/4) # 4 i per input sample. 

            sample, generated_binary_mask, generated_grade_mask, modality_label = inpaint_tuple

            sample.save(os.path.join(output_path, f"inpaint_{inpaint_num}_input_{input_num}_{modality_label}_sample.png"))
            generated_binary_mask.save(os.path.join(output_path, f"inpaint_{inpaint_num}_input_{input_num}_tumor_binary_mask.png"))
            generated_grade_mask.save(os.path.join(output_path, f"inpaint_{inpaint_num}_input_{input_num}_tumor_grade_mask.png"))

        logging.debug(f"Saved all inpaints to {output_path}")

    except Exception as e:
        logging.error(
            f"Error while trying to save generated inpaints in {output_path}: {e}"
        )
        raise e



def generate(
    model_file:str = 'inp_gen.pth', 
    num_samples: int = 10, 
    save_images: bool = True, 
    output_path: str = 'testing/', 
    image_size:tuple = (256, 256), 
    num_inpaints_per_sample:int=10, 
    randomize_input_image_order:bool=True,
    F_img_path:str=None, 
    T1_img_path:str=None, 
    T1c_img_path:str=None, 
    T2_img_path:str=None,
    add_variations_to_mask:bool=True,
    x_center:float = 130.,
    y_center:float = 130., 
    radius_1:float = 10., 
    radius_2:float = 15.,
    radius_3:float = 30.,
    ):
    """Generating and returning inpaints for Flair, T1, T1 contrast enhanced, T2. based on tumor grade masks created via concentric circles. """

    try:
        logging.debug("Generating inpaints...")

        # Using default image size: Check and transform to tuple if necessary
        if isinstance(image_size, int) or isinstance(image_size, float):
            image_size = (int(image_size), int(image_size))

        elif isinstance(image_size, list) or isinstance(image_size, tuple):
            if len(image_size) > 2:
                logging.warning(
                    f"You provided an image_size of {image_size} (type: {type(image_size)}) of length != 2. Please note that output of this model are two-dimensional grayscale images. Only values of image_size at index position 0 and 1 will be used in the following."
                )
            elif len(image_size) < 2:
                image_size = (image_size[0], image_size[0])
            image_size = (int(image_size[0]), int(image_size[1]))
        else:
            raise Exception(
                f"image_size needs to be of type int, list or tuple. You provided type={type(image_size)}: {image_size}."
            )
        
        if image_size[0] != 256:
            logging.warning(
                f"You provided an image_size[0]={image_size[0]}, which is != 256. "
                f"This might affect the output inpaints as =256 is default. Please revise image_size {image_size} and, if need be, set to default: tuple of (256, 256)."
            )
        
        # Init models in case a different model file was provided i.e. inp_gen_path!='./inp_gen.pth'
        init_models(inp_gen_path=model_file)

        # Init image list based on Flair, T1, T1c, T2 MRI image folders
        image_list_matrix = [F_img_path, T1_img_path, T1c_img_path, T2_img_path]
        for i, input_path in enumerate(image_list_matrix):
            if input_path is not None:
                if os.path.isfile(input_path) and input_path.endswith('.png'):
                    try:
                        image_list = glob.glob(input_path)
                    except:
                        # allow relative links
                        image_list = glob.glob(model_root_path + "/" + input_path)
                    image_list_matrix[i] = image_list
                elif os.path.isdir(input_path):
                    if input_path.endswith('/'):
                        try:
                            image_list = glob.glob(input_path + '*.png')
                        except:
                            # allow relative links
                            image_list = glob.glob(model_root_path + "/" + input_path + '*.png')

                        image_list_matrix[i] = image_list
                    else:
                        try:
                            image_list = glob.glob(input_path + '/*.png')
                        except:
                            # allow relative links
                            image_list = glob.glob(model_root_path + "/" + input_path + '/*.png')
                        image_list_matrix[i] = image_list
                else:
                    logging.warning(
                        f"You provided an input_path that neither points to a folder with png images nor to a png image: {input_path}."
                        f"Please revise and make sure access to the folder exists. Instead of using your input_path, we now proceed to generate images from default examples in 'temp_slice/'."
                    )
            else:
                if i == 0: 
                    image_list_matrix[i] = F_default_img_list
                elif i ==1:
                    image_list_matrix[i] = T1_default_img_list
                elif i ==2:
                    image_list_matrix[i] = T1c_default_img_list
                elif i ==3:
                    image_list_matrix[i] = T2_default_img_list


        # Check if num_samples > image_list with least entries. Issue warning that the same image will be used multiple times
        max_index = 9999999
        for i, image_list in enumerate(image_list_matrix):
            if len(image_list) <= num_samples:
                if i == 0: 
                    image_list_name = "F_default_img_list"
                elif i ==1:
                    image_list_name = "T1_default_img_list"
                elif i ==2:
                    image_list_name = "T1c_default_img_list"
                elif i ==3:
                    image_list_name = "T2_default_img_list"
                if max_index >= len(image_list):
                    max_index=len(image_list)-1
                logging.warning(
                    f"Your specified number of images to be generated (num_samples={num_samples}) is greater than the number of images in at least one of the image lists of the 4 input image lists F, T1, T1c, T2: {image_list_name}, length={len(image_list)}. "
                    f"During image generation, the same image will be used multiple times: The number of different images used for inpainting will be = {max_index}. Note: You can also increase the number of generated images by increasing the number of inpaints per input image (currently: num_inpaints_per_sample is set to {num_inpaints_per_sample})."
                )

        output_list = get_inpaints(
            num_samples=num_samples,
            image_size=image_size,
            F_img_list = image_list_matrix[0],
            T1_img_list = image_list_matrix[1],
            T1c_img_list = image_list_matrix[2],
            T2_img_list = image_list_matrix[3],
            max_index=max_index,
            num_inpaints_per_sample=num_inpaints_per_sample,
            randomize_input_image_order=randomize_input_image_order,
            add_variations_to_mask=add_variations_to_mask,
            x_center = x_center,
            y_center = y_center, 
            radius_1 = radius_1, 
            radius_2 = radius_2, 
            radius_3 = radius_3,
        )

        if save_images:
            save_generated_inpaints(
                output_list=output_list, 
                output_path=output_path, 
                num_inpaints_per_sample=num_inpaints_per_sample,
        )
        else:
            return output_list
    except Exception as e:
        logging.error(f"Error while trying to generate from {num_samples} input samples each of which generates {num_inpaints_per_sample} inpaints: {e}")
        raise e
