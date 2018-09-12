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
from skimage.color import rgb2hsv,rgb2lab
#####################################################
## functions
#####################################################
#read video file frame by frame, beginning and ending with a timestamp
def read_video_segments(video,start_frame,end_frame,resolution_width=200,target_colorspace=None):
    resolution_height=int(round(resolution_width * 9/16))
    resolution=(resolution_width,resolution_height)
    vid = cv2.VideoCapture(video)
    frames=[]
    vid_length=0
    with tqdm(total=end_frame-start_frame+1) as pbar: #init the progressbar,with max lenght of the given segment
        while(vid.isOpened()):
            # Capture frame-by-frame
            ret, frame = vid.read() # if ret is false, frame has no content
            if not ret:
                break
            # skip every "skip_frame"
            if vid_length>=start_frame:
                # resize the video to a different resolution
                frame=cv2.resize(frame,resolution)
                frames.append(frame) #add the individual frames to a list
                pbar.update(1) #update the progressbar
            if vid_length==end_frame:
                pbar.update(1)
                break
            vid_length+=1 #increase the vid_length counter
    vid.release()
    cv2.destroyAllWindows()
    frames=change_colorspace(frames,target_colorspace)
    return frames
##################################################
def change_colorspace(frame_list,target_colorspace):
    print(target_colorspace)
    changed_frame_list=[]
    if target_colorspace=='HSV':
        changed_frame_list = [rgb2hsv(frame) for frame in frame_list]
        return changed_frame_list
    if target_colorspace=='cie-lab':
        changed_frame_list = [rgb2lab(frame) for frame in frame_list]
        return changed_frame_list
    else:
        return frame_list
##################################################
def extract_dominant_colors(frame_list):
    print(str(len(frame_list))+' frames to process.')
    rgb_to_color=fn_rgb_to_color() #get the color dict 
    bins={} #bins dict for histograms 
    for rgb in rgb_to_color: #init the dict with zeros for every key
        bins[rgb_to_color[rgb]]=0
    rgb_list=[] #create a traverseable list of the rgb_values
    for rgb in rgb_to_color: #map the values of the dict to a list
        rgb_list.append(rgb)
    i = 0

    kdt = KDTree(rgb_list, leaf_size=30, metric='euclidean')  
    for image in tqdm(frame_list): #traverse the video
        img = image.reshape((image.shape[0] * image.shape[1], 3)) #flatten the image to 1d   
        nns = kdt.query(img, k=1, return_distance=False)
        for nn in nns:
            bins[rgb_to_color[rgb_list[nn[0]]]]+=1
        i+=1
    norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1] #normalize the binsi
    bins_norm={k:v/norm_factor for k,v in bins.items()}
    return bins_norm
####################################################
def bins_to_df(bins,bin_threshold=5,colors_to_return=5):
    #create a dataframe, sorted descending by count
    bins_sorted=sorted(zip(list(bins.values()),list(bins.keys())),reverse=True)
    df=pd.DataFrame(bins_sorted,columns=['count','color'])
    df.set_index('color',inplace=True) #set the colors as the index of the dataframe
    bin_threshold=bin_threshold/100 #scale the percentage to 0-1
    df = df[df>bin_threshold].dropna() #kick bins from the dataframe with precentage lower than bin_threshold 
    return df.head(colors_to_return)#return the color_return highest bins, default 5, if less bins then
                                    #color_return are there return all
#####################################################
def fn_rgb_to_color(*path):
    if not ('no'):
        path=str(path)[2:-3] #to get rid of the of the *args things
        rgb_to_color = {}
        with open(path) as f:
            for line in f:
                #split lines at "::
                color, rgb = line.strip().split(':')
                #strip the rgb-string of the parenthesis, split it up a the commas,
                #cast them to int and put them into a tuples
                rgb_value=tuple(map(int,(rgb.strip('(').strip(')').split(','))))
                rgb_to_color[rgb_value] = color
    else:
        colors={'darkred':(139,0,0),
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
        'grey':(28,128,128),
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
        if args.target_colorspace=='HSV':
            print('HSV')
            for color in colors:
                colors_aux[color]=tuple(rgb2hsv(np.array((colors[color])).reshape(1,1,3)).reshape(3))
            colors=colors_aux
        if args.target_colorspace=='cie-lab':
            print('cie-lab')
            for color in colors:
#                 print(tuple(rgb2lab(np.array((colors[color])).reshape(1,1,3)).reshape(3)))
                colors_aux[color]=tuple(rgb2lab(np.array((colors[color])).reshape(1,1,3)).reshape(3))
            colors=colors_aux
        rgb_to_color={}
        for color in colors:
            rgb_to_color[colors[color]]=color
        #purple4 is median purple
        #skin is caucasian        
    return rgb_to_color
######################################################
def read_azp(azp_path):
    #extract the .azp-file to /tmp
    zip_ref = zipfile.ZipFile(azp_path)
    zip_ref.extractall('/tmp')
    #read the .xml-file
    tree = ET.parse('/tmp/content.xml')
    root = tree.getroot().findall('./{http://experience.univ-lyon1.fr/advene/ns}annotations')
    #traverse the .xml-file
    with open(args.output_path,'w') as file:
            if args.what_to_process=='scene':
                segment_list=[]
            for child in root[0].iter():
                if child.get('type')=='#Shot': #whenever a shot annotation is found, extract the timestamp from the xml
                    dominant_colors_list=[]
                    for child2 in child:
                        if child2.tag=='{http://experience.univ-lyon1.fr/advene/ns}millisecond-fragment':
                            end=round(int(child2.get('end'))/1000*25) #timestamps are rounded, because there are no half frames
                            begin=round(int(child2.get('begin'))/1000*25)
                            if args.what_to_process=='scene': #if 'scene' is selected append the frames of the segments to a list
                                segment_list.append(read_video_segments(args.video_path,begin,end,args.resolution_width,args.target_colorspace))
                            if args.what_to_process=='segment': #if 'segment' is selected run extract_dominant_colors on the segment
                                segment = read_video_segments(args.video_path,begin,end,args.resolution_width,args.target_colorspace)
                                colors_df = bins_to_df(extract_dominant_colors(segment),args.bin_threshold,args.colors_to_return)
                                colors_list = [(color,perc) for color,perc in zip(colors_df.index.values,colors_df.values.tolist())]
                                print(begin,end,colors_list)
                                file.write((begin,end,colors_list)+'\n') #write the timestamp and the extracted colors to file
            if args.what_to_process=='scene': #if 'scene' is selected run extract_dominant_colors on the the list of segments
                colors_df = bins_to_df(extract_dominant_colors(segment_list),args.bin_threshold,args.colors_to_return)
                colors_list = [(color,perc) for color,perc in zip(colors_df.index.values,colors_df.values.tolist())]
                print(colors_list)
                file.write(colors_list+'\n') #write the extracted colors to file
            file.close()
######################################################
def azp_path(path):
    if path[-4:] == '.azp': #if the path is to a single file
        print('exactly')
        read_azp(path)
    elif path[0][-4:] == '.azp': #if the path is to several files
        print('like')
        for azp_path in path:
            #print(azp_path)
            read_azp(azp_path)
    else: #else it is assumed the path points to a directory
        directory_content = os.listdir(path)
        azp_list=[]
        for elem in directory_content:
            if elem[-4:]=='.azp':
                if path[-1]=='/': #if the path ends with an '/', add the .azp-file
                    azp_list.append(path+elem)    
                else: #else, add a '/' and then the .azp-file
                    azp_list.append(path+'/'+elem)
        for azp_path in azp_list:
            read_azp(azp_path)
        print('planned')
######################################################
if __name__ == "__main__":
        ##############################################
        ## command line arguments
        ##############################################
        #arguments for command line
        parser = argparse.ArgumentParser()
        parser.add_argument("video_path",help="the path to the videofile")
        parser.add_argument("azp_path",help="the path to a azp-file, a list of .azp-files or the path to a directory cotaining .azp-files")
        #optional arguments
        parser.add_argument("--output_path",nargs='?',default='dominant_colors.txt',help="optional,the path for the output .txt-file that should contain the dominant colors, has to include the filename as a .txt-file,default = dominant_colors.txt")       
        parser.add_argument("--resolution_width",type=int,nargs='?',default=200,help="optional, set the resolution width of the videofile, the resolution scales automatically to 16:9,default = 200")
        parser.add_argument("--bin_threshold",type=float,nargs='?',default=5, help="optional, set the percentage (0-100) a color has to reach to be returned,default = 5")
        parser.add_argument("--colors_to_return",type=int,nargs='?',default=5, help="optional, set how many colors should be returned at maximum,default = 5")
        parser.add_argument("--colors_txt",nargs='?', help="optional, path to a .txt-file containing colors, the file must be in the format 'black:(0,0,0) new line red:(255,0,0) etc',default are a list of 40 colors hardcoded")
        parser.add_argument("--what_to_process",nargs='?',default='segment',help="optional,decide if the dominant colors should be processed per segment or a whole scene, default is segment, switch to scene with 'scene'")
        parser.add_argument("--target_colorspace",nargs='?',help='change the colorspace of the video, for now only supports HSV and cie-lab')
        args=parser.parse_args()
        ##############################################
        ## main
        ##############################################
        azp_path(args.azp_path)
        print('done')
