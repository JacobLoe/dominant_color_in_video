#####################################################
## libraries
#####################################################
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import time
import argparse
#####################################################
## methods
#####################################################
#read video file frame by frame
def read_video(video,skip_frames,resolution):
    res_dict={'1':(120,90),'2':(240,135),'3':(480,270)}
    vid = cv2.VideoCapture(video)
    frames=[]
    vid_length=0
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read() # if ret is false, frame has no content
        # resize the video to a different resolution
        if ret:
            frame=cv2.resize(frame,res_dict[resolution])
        # skip every "skip_frame"
        if vid_length%skip_frames==0:
            frames.append(frame) #add the individual frames to a list
        vid_length+=1 #increase the vid_length counter
        if not ret:
            break
    vid.release()
    cv2.destroyAllWindows()
    return frames
###############################################
def extract_dominant_color(frame_list,rgb_to_color):
    start=time.time()
    #bins dict for histograms
    bins={}
    rgb_list=[]
    #init the dict with zeros for every key
    for rgb in rgb_to_color:
        bins[rgb_to_color[rgb]]=0
    #map the values of the dict to a list
    for rgb in rgb_to_color:
        rgb_list.append(rgb)
    i = 0
    for image in frame_list:
        #flatten the image to 1d 
        img = image.reshape((image.shape[0] * image.shape[1], 3))     

        #nearest neighbour search
        for pixel in img:
            bin_aux=[]
            #get the euclidean distance between the colors and the current pixel
            for rgb in rgb_list:
                bin_aux.append(euclidean(pixel,rgb))
            # get the index of the color,which has the smallest distance, in rgb_list
            min_pos = np.argmin(bin_aux)
            #increment the respective color 
            bins[rgb_to_color[rgb_list[min_pos]]]+=1
        i+=1
        end=time.time()
        print('Finished '+str(i)+',time: '+str(end-start))
    #create a dataframe, sorted by count and return it
    bins_sorted=sorted(zip(list(bins.values()),list(bins.keys())),reverse=True)
    df=pd.DataFrame(bins_sorted,columns=['count','color'])
    df.set_index('color',inplace=True) #set the colors as the index of the dataframe
    return df
###################################################
#
def fn_rgb_to_color():
	colors={'darkred':(139,0,0),
	'firebrick':(178,34,34),
	'crimson':(220,20,60),
	'red':(255,0,0),
	'tomato':(255,99,71),
	'salmon':(250,128,114),
	'dark_orange':(255,140,0),
	'gold':(255,215,0),
	'dark_khaki':(189,183,107),
	'yellow':(255,255,0),
	'dark_olive_green':(85,107,47),
	'olive_drab':(107,142,35),
	'green_yellow':(173,255,47),
	'dark_green':(0,100,0),
	'aqua_marine':(127,255,212),
	'steel_blue':(70,130,180),
	'sky_blue':(135,206,235),
	'dark_blue':(0,0,139),
	'blue':(0,0,255),
	'royal_blue':(65,105,225),
	'purple':(128,0,128),
	'violet':(238,130,238),
	'deep_pink':(255,20,147),
	'pink':(255,192,203),
	'antique_white':(250,235,215),
	'saddle_brown':(139,69,19),
	'sandy_brown':(244,164,96),
	'ivory':(255,255,240),
	'dim_grey':(105,105,105),
	'grey':(28,128,128),
	'silver':(192,192,192),
	'light_grey':(211,211,211),
	'black':(0,0,0),
	'white':(255,255,255),
	'dark_cyan':(0,139,139),
	'cyan':(0,255,255),
	'green':(0,128,0),
	'khaki':(240,230,140),
	'golden_rod':(218,165,32),
	'orange':(255,165,0),
	'coral':(255,127,80),
	'magenta':(255,0,255),
	'wheat':(245,222,179),
	'skin':(255,224,189),
	'purple4':(147,112,219)}
	rgb_to_color={}
	for color in colors:
	    rgb_to_color[colors[color]]=color
	#purple4 is median purple
	#skin is caucasian 
	return rgb_to_color
##############################################
## main
##############################################

parser = argparse.ArgumentParser()
parser.add_argument("path",help="the path to the videofile")
parser.add_argument("skip_frames",help="skip every n-th frame in the videofile",type=int)
parser.add_argument("resolution",help="set the resolution of the videofile, '1':(120,90),'2':(240,135),'3':(480,270)")
args=parser.parse_args()

frame_list = read_video(args.path,args.skip_frames,args.resolution)
df = extract_dominant_color(frame_list,fn_rgb_to_color())
print(df)
