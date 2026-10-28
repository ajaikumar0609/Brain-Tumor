#! /bin/bash

echo "Running a test: Generate method of this medigan model module."

echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. Pip install dependencies from requirements.txt"
pip install --upgrade pip
pip3 install --timeout 1000 torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt

echo "3. Run the generate function with parameters"

python -c "from __init__ import generate; 

model_file:str = 'MODEL/inp_gen.pth'
num_samples: int = 3
save_images: bool = True
output_path: str = 'examples/' 
image_size:tuple = (256, 256) 
num_inpaints_per_sample:int=5
randomize_input_image_order:bool=True
F_img_path:str= 'Norm_slice/F/'
T1_img_path:str=None
T1c_img_path:str=None 
T2_img_path:str=None
add_variations_to_mask:bool=True
x_center:float = 131.
y_center:float = 131. 
radius_1:float = 11.
radius_2:float = 16.
radius_3:float = 31.

generate(
    model_file = model_file, 
    num_samples= num_samples,
    save_images=save_images,
    output_path=output_path,
    image_size=image_size,
    num_inpaints_per_sample=num_inpaints_per_sample,
    randomize_input_image_order=randomize_input_image_order,
    F_img_path=F_img_path,
    T1_img_path=T1_img_path,
    T1c_img_path=T1c_img_path,
    T2_img_path=T2_img_path,
    add_variations_to_mask=add_variations_to_mask,
    x_center=x_center,
    y_center=y_center,
    radius_1=radius_1,
    radius_2=radius_2,
    radius_3=radius_3,
    )"
echo "4. Any errors? Have synthetic images been successfully created in folder /examples?"
