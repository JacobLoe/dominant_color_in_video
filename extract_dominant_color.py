#####################################################
## libraries
#####################################################
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import time
import argparse
import zipfile
import xml.etree.ElementTree as ET
from tqdm import tqdm
import os
from scipy.spatial.distance import euclidean
#####################################################
## functions
#####################################################
#read video file frame by frame, beginning and ending with a timestamp
def read_video(video,start_frame,end_frame,resolution_width,target_colorspace):
    resolution_height=int(round(resolution_width * 9/16))
    resolution=(resolution_width,resolution_height)
    vid = cv2.VideoCapture(video)
    frames=[]
    vid_length=0
    with tqdm(total=end_frame-start_frame+1) as pbar: #init the progressbar,with max length of the given segment
        while(vid.isOpened()):
            ret, frame = vid.read() # if ret is false, frame has no content
            if not ret:
                break
            if vid_length>=start_frame:
                frame=cv2.resize(frame,resolution) # resize the video to a different resolution
                frame=np.array(frame,dtype='float32')
                frames.append(frame) #add the individual frames to a list
                pbar.update(1) #update the progressbar
            if vid_length==end_frame:
                pbar.update(1)
                break
            vid_length+=1 #increase the vid_length counter
    vid.release()
    cv2.destroyAllWindows()
    frames=change_colorspace(frames,target_colorspace)
    return frames[:-1]
##################################################
def change_colorspace(frame_list,target_colorspace):
    changed_frame_list=[]
    if target_colorspace=='HSV':
        print('HSV')
        for frame in frame_list:
            frame=np.array(frame,dtype='uint8')
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame=np.array(frame,dtype='float32')
            changed_frame_list.append(frame)
        return changed_frame_list
    if target_colorspace=='cie-lab':
        print('cie-lab')
        for frame in frame_list:
            frame=np.array(frame,dtype='uint8')
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame=np.array(frame,dtype='float32')
            changed_frame_list.append(frame)
        return changed_frame_list
    else:
        print('rgb')
        for frame in frame_list:
            frame=np.array(frame,dtype='uint8')
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame=np.array(frame,dtype='float32')
            changed_frame_list.append(frame)
        return changed_frame_list
##################################################
def extract_dominant_colors(frame_list,target_colorspace,path,colors_to_return=5):
    value_to_color=fn_value_to_color(target_colorspace,path) #get the color dict 

    bins={} #bins dict for histogram
    for value in value_to_color: #init the dict with ones for every key to avoid difficulties with divisions
                            # because the the sum of the bins goes from 500k to 2kk this shouldn't be a problem
        bins[value_to_color[value]]=1

    color_value_list=[value for value in value_to_color] #create a list of the color_values

    kdt = KDTree(color_value_list, leaf_size=30, metric='euclidean')
    for image in tqdm(frame_list): #traverse the video
        img = image.reshape((image.shape[0] * image.shape[1], 3)) #flatten the image to 1d   
        nns = kdt.query(img, k=1, return_distance=False)
        for nn in nns:
            bins[value_to_color[color_value_list[nn[0]]]]+=1

    norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1] #normalize the bins
    bins_norm={k:v/norm_factor*100 for k,v in bins.items()}
    bins_sorted=sorted(zip(list(bins_norm.values()),list(bins_norm.keys())),reverse=True)
    return bins_sorted[:colors_to_return]
#####################################################
def fn_value_to_color(target_colorspace,path):
            if (path != 'full'):
                print('Now using colors specified in path')
                colors_to_value_dict = {}
                with open(path) as f:
                    for line in f:
                        #split lines at "::
                        color, rgb = line.strip().split(':')
                        #strip the rgb-string of the parenthesis, split it up a the commas,
                        #cast them to int and put them into a tuples
                        rgb_value=tuple(map(int,(rgb.strip('(').strip(')').split(','))))
                        colors_to_value_dict[color]=rgb_value
            else:
                colors_to_value_dict={
                'darkred':(139,0,0),
                'firebrick':(178,34,34),
                'crimson':(220,20,60),
                'red':(255,0,0),
                'tomato':(255,99,71),
                'salmon':(250,128,114),
                'darkorange':(255,140,0),
                'gold':(255,215,0),
                'darkkhaki':(189,183,107),
                'yellow':(255,255,0),
                'darkolivegreen':(85,107,47),
                'olivedrab':(107,142,35),
                'greenyellow':(173,255,47),
                'darkgreen':(0,100,0),
                'aquamarine':(127,255,212),
                'steelblue':(70,130,180),
                'skyblue':(135,206,235),
                'darkblue':(0,0,139),
                'blue':(0,0,255),
                'royalblue':(65,105,225),
                'purple':(128,0,128),
                'violet':(238,130,238),
                'deeppink':(255,20,147),
                'pink':(255,192,203),
                'antiquewhite':(250,235,215),
                'saddlebrown':(139,69,19),
                'sandybrown':(244,164,96),
                'ivory':(255,255,240),
                'dimgrey':(105,105,105),
                'grey':(128,128,128),
                'silver':(192,192,192),
                'lightgrey':(211,211,211),
                'black':(0,0,0),
                'white':(255,255,255),
                'darkcyan':(0,139,139),
                'cyan':(0,255,255),
                'green':(0,128,0),
                'khaki':(240,230,140),
                'goldenrod':(218,165,32),
                'orange':(255,165,0),
                'coral':(255,127,80),
                'magenta':(255,0,255),
                'wheat':(245,222,179),
                'skin':(255,224,189),
                'purple4':(147,112,219)}

            colors_aux={}
            if target_colorspace=='HSV':
                print('HSV')
                for color in colors_to_value_dict:
                    a = np.array((colors_to_value_dict[color]),dtype='uint8')
                    b = a.reshape(1,1,3)
                    c = cv2.cvtColor(b,cv2.COLOR_RGB2HSV)
                    c=np.array(c,dtype='float32')
                    colors_aux[color]=tuple(c.reshape(3))
                colors_to_value_dict=colors_aux

            if target_colorspace=='cie-lab':
                print('cie-lab')
                for color in colors_to_value_dict:
                    a = np.array((colors_to_value_dict[color]),dtype='uint8')
                    b = a.reshape(1,1,3)
                    c = cv2.cvtColor(b,cv2.COLOR_RGB2LAB)
                    c=np.array(c,dtype='float32')
                    colors_aux[color]=tuple(c.reshape(3))
                colors_to_value_dict=colors_aux

            value_to_color={}
            for color in colors_to_value_dict:
                value_to_color[colors_to_value_dict[color]]=color
            #purple4 is median purple
            #skin is caucasian        
            return value_to_color
######################################################
if __name__ == "__main__":
    ## command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path",help="the path to the videofile")
    parser.add_argument("azp_path",help="the path to a azp-file")
    #optional arguments
    parser.add_argument("--output_path",nargs='?',default='dominant_colors.txt',help="optional,the path for the output .txt-file that should contain the dominant colors, has to include the filename as a .txt-file,default = dominant_colors.txt")       
    parser.add_argument("--resolution_width",type=int,nargs='?',default=200,help="optional, set the resolution width of the videofile, the resolution scales automatically to 16:9,default = 200")
    parser.add_argument("--colors_to_return",type=int,nargs='?',default=5, help="optional, set how many colors should be returned at maximum,default = 5")
    parser.add_argument("--colors_txt",nargs='?',default='full', help="optional, path to a .txt-file containing colors, the file must be in the format 'black:(0,0,0) new line red:(255,0,0) etc',default are a list of 40 colors hardcoded")
    parser.add_argument("--target_colorspace",nargs='?',default='cie-lab',help='change the colorspace of the video, for now only supports rgb,HSV and cie-lab,default is cie-lab')
    args=parser.parse_args()
    ##############################################
    #extract the .azp-file to /tmp
    zip_ref = zipfile.ZipFile(args.azp_path)
    zip_ref.extractall('/tmp')
    #read the .xml-file
    tree = ET.parse('/tmp/content.xml')
    root = tree.getroot().findall('./{http://experience.univ-lyon1.fr/advene/ns}annotations')
    #traverse the .xml-file
    with open(args.output_path,'w') as file:
            for child in root[0].iter():
                if child.get('type')=='#Shot': #whenever a shot annotation is found, extract the timestamp from the xml
                    for child2 in child:
                        if child2.tag=='{http://experience.univ-lyon1.fr/advene/ns}millisecond-fragment':
                           end=round(int(child2.get('end'))/1000*25) #timestamps are rounded, because there are no half frames
                           begin=round(int(child2.get('begin'))/1000*25)
                           try:
                               frame_list = read_video(args.video_path,begin,end,args.resolution_width,args.target_colorspace)
                               dominant_colors = extract_dominant_colors(frame_list,args.target_colorspace,args.colors_txt,args.colors_to_return)

                               print(str(child2.get('begin')),str(child2.get('end')),dominant_colors)
                               file.write(str(child2.get('begin'))+' '+str(child2.get('end'))+' '+str(dominant_colors)+'\n') #write the timestamp and the extracted colors to file
                           except:
                               print('timestamp not possible')
                               pass
    file.close()
