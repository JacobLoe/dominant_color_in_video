#####################################################
## libraries
#####################################################
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KDTree
import time
import argparse
import json

#####################################################
## methods
#####################################################
#read video file frame by frame, beginning and ending with a timestamp
def read_video_segments(video,start_frame,end_frame,resolution_width):
    resolution_height=int(round(resolution_width * 9/16))
    resolution=(resolution_width,resolution_height)
    vid = cv2.VideoCapture(video)
    frames=[]
    vid_length=0
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
        if vid_length==end_frame:
            break
        vid_length+=1 #increase the vid_length counter
    vid.release()
    cv2.destroyAllWindows()
    return frames
###############################################
def extract_dominant_color(frame_list,bin_threshold=0.05,colors_to_return=5):
    print(str(len(frame_list))+' frames to process.')
    start=time.time()
    rgb_to_color=fn_rgb_to_color() #get the color dict 
    bins={} #bins dict for histograms 
    for rgb in rgb_to_color: #init the dict with zeros for every key
        bins[rgb_to_color[rgb]]=0
    rgb_list=[] #create a traverseable list of the rgb_values
    for rgb in rgb_to_color: #map the values of the dict to a list
        rgb_list.append(rgb)
    i = 0

    kdt = KDTree(rgb_list, leaf_size=30, metric='euclidean')  
    for image in frame_list: #traverse the video
        #flatten the image to 1d 
        img = image.reshape((image.shape[0] * image.shape[1], 3))     
        nns = kdt.query(img, k=1, return_distance=False)
        for nn in nns:
            bins[rgb_to_color[rgb_list[nn[0]]]]+=1
#        for pixel in img: # do nearest neighbour search on every pixel every color in the list
#            bin_aux=[]
#            #get the euclidean distance between the colors and the current pixel
#            for rgb in rgb_list:
#                bin_aux.append(euclidean(pixel,rgb))
#            # get the index of the color,which has the smallest distance, in rgb_list
#            min_pos = np.argmin(bin_aux)
#            #increment the respective color 
#            bins[rgb_to_color[rgb_list[min_pos]]]+=1
        i+=1
        end=time.time()
        print('Finished '+str(i)+',time: '+str(end-start))
    #create a dataframe, sorted descending by count
    bins_sorted=sorted(zip(list(bins.values()),list(bins.keys())),reverse=True)
    df=pd.DataFrame(bins_sorted,columns=['count','color'])
    df.set_index('color',inplace=True) #set the colors as the index of the dataframe
    norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1]  #normalize the bins
    df=df/norm_factor
    bin_threshold=bin_threshold/100 #scale the percentage to 0-1
    df = df[df>bin_threshold].dropna() #kick bins from the dataframe with precentage lower than bin_threshold 
    return df.head(colors_to_return)#return the color_return highest bins, default 5, if less bins then
                                #color_return are there return all
###################################################
#
def fn_rgb_to_color():
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
	rgb_to_color={}
	for color in colors:
	    rgb_to_color[colors[color]]=color
	#purple4 is median purple
	#skin is caucasian 
	return rgb_to_color
##############################################
## command line arguments
##############################################

#arguments for command line
parser = argparse.ArgumentParser()
parser.add_argument("video_path",help="the path to the videofile")
parser.add_argument("json_path",help="the path to the json-file")
parser.add_argument("resolution_width",type=int,help="set the resolution width of the videofile, the resolution scales automatically to 16:9")
parser.add_argument("bin_threshold",type=float,help="set the percentage (0-100) a color has to reach to be returned,default 5")
parser.add_argument("colors_to_return",type=int,help="set how many colors should be returned at maximum,default 5")
args=parser.parse_args()

##############################################
## main
##############################################
# read json file 
with open(args.json_path) as file:
    ground_truth=json.load(file)
    file.close()
# extract the start and end of the timestamps and enter them in a list 
gt_e_s_list =[]

for i,content in enumerate(ground_truth['annotations']):
    print ("{0}/{1}".format(i+1, len(ground_truth['annotations'])))
    gt_e_s=[]
    gt_e_s.append(round(content['begin']/1000*25))
    gt_e_s.append(round(content['end']/1000*25))
    gt_e_s_list.append(gt_e_s)

#add all segments to a list
segment_list=[]
for i,gt_i in enumerate(gt_e_s_list):
    print ("{0}/{1}".format(i+1, len(gt_e_s_list)))
    segment_list.append(read_video_segments(args.video_path,
                    gt_i[0],gt_i[1],args.resolution_width))
#extract dominant colors for each segment
dominant_colors_list=[]
for segment in segment_list:
    dominant_colors_list.append(extract_dominant_color(segment,
		args.bin_threshold,args.colors_to_return))
#write the predictions to a dict and then to a .json-file
predictions=ground_truth
for i,segment in enumerate(predictions['annotations']):
    segment['content']=dominant_colors_list[i].index.tolist()
predictions_json=json.dumps(predictions)
with open('predictions.json','w') as json_file:
    json_file.write(predictions_json)
    json_file.close()
print('done')
