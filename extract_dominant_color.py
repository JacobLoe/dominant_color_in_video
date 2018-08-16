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
import zipfile
import xml.etree.ElementTree as ET
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
def extract_dominant_color(frame_list):
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
        i+=1
        end=time.time()
        print('Finished '+str(i)+',time: '+str(end-start))
        norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1]#normalize the bins
        bins_norm={k:v/norm_factor for k,v in bins.items()}
    return bins_norm
###################################################
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

if __name__ == "__main__":
	##############################################
	## command line arguments
	##############################################
	#arguments for command line
	parser = argparse.ArgumentParser()
	parser.add_argument("video_path",help="the path to the videofile")
	parser.add_argument("azp_path",help="the path to the azp-file")
	parser.add_argument("resolution_width",type=int,help="set the resolution width of the videofile, the resolution scales automatically to 16:9")
	parser.add_argument("bin_threshold",type=float,help="set the percentage (0-100) a color has to reach to be returned,default 5")
	parser.add_argument("colors_to_return",type=int,help="set how many colors should be returned at maximum,default 5")
	args=parser.parse_args()
	##############################################
	## main
	##############################################
	#extract the .azp-file	
	zip_ref = zipfile.ZipFile(args.azp_path)
	zip_ref.extractall('zip')
	#read the .xml-file
	tree = ET.parse('zip/content.xml')
	root = tree.getroot().findall('./{http://experience.univ-lyon1.fr/advene/ns}annotations')
	#traverse the .xml-file
	for child in root[0].iter():
	    #whenever a shot annotation is found, start color extraction 
	    if child.get('type')=='#Shot':
		dominant_colors_list=[]
		for child2 in child:
		    if child2.tag=='{http://experience.univ-lyon1.fr/advene/ns}millisecond-fragment':
		        begin=int(child2.get('begin'))/1000*25
		        end=int(child2.get('end'))/1000*25
			segment = read_video_segments(args.video_path,
				begin,end,
				args.resolution_width)
			dominant_colors_list.append(begin,
					end,
					extract_dominant_colors(
					segment,
					args.bin_threshold,
					args.colors_to_return))


	print('done')
