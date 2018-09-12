### installation
pip3 install pandas
pip3 install tqdm
pip3 install sklearn
pip3 install scikit-image
pip3 install numpy
pip3 install 

pip3 install nose

### manual

python3 extract_dominant_color.py video_path azp_path

#### arguments

video_path: the path to the videofile

azp_path: the path to a azp-file, a list of .azp-files or the path to a directory cotaining .azp-files

##### optional arguments

output_path: the path for the output .txt-file that should contain the dominant colors, has to include the filename as a .txt-file,default = dominant_colors.txt

resolution_width: set the resolution width of the videofile, the resolution scales automatically to 16:9,default = 200

bin_threshold: set the percentage (0-100) a color has to reach to be returned,default = 5

colors_to_return: set how many colors should be returned at maximum,default = 5

colors_txt: path to a .txt-file containing colors, the file must be in the format 'black:(0,0,0) new line red:(255,0,0) etc',default are a list of 40 colors hardcoded

what_to_process: decide if the dominant colors should be processed per segment or a whole scene, default is segment, switch to scene with 'scene'

target_colorspace: change the colorspace of the video, for now only supports HSV and cie-lab


