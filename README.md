# Style Transfer Model Script for Batches of Images
## Installation
* get python 3.7 or later
* open cmd prompt or powershell in the directory you cloned the project to
* run `style-env/Scripts/activate.bat`
* run `pip install numpy TensorFlow tensorflow_hub pillow`
* add your input files to `./input` or whatever `input_directory` is set to (the program will filter by file type, so select what filetype you want by setting `input_filetype`, default is `.png`)
* add your style photo as style.jpg in the main directory
* run `python main.py` (this will spit out a few warnings about not having a GPU configured, these are fine. it will take a few seconds to a few minutes depending on how many images you have to process. if it's taking to long you can change the `image_res` variable in the python program to be something smaller)
* check `./output` or whatever `output_directory` is set to for the output images

## Useful Python Program Modifications
### `image_res = [1920, 1080]`
Default value is for 1080p input images. Input images are scaled down from 1080p to 720p during processing by default.

### `input_filetype = '.png'`
Sets the file type of expected input images. Files of other types in the input directory will be ignored.

### `input_directory = './input/'`
Sets the directory of input images.

### `output_directory = './output/'`
Sets the directory of output images.

### `style_location = './style.jpg'`
Location of the image to be used as the style reference.