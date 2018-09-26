#####################################################
## libraries
#####################################################
import zipfile
import xml.etree.ElementTree as ET
import extract_dominant_color as edc
import argparse
import numpy as np
import cv2
import os
######################################################
if __name__ == "__main__":
        ##############################################
        parser = argparse.ArgumentParser()
        parser.add_argument("azp_path",help="the path to a azp-file")
        parser.add_argument("video_path",help="the path to a video")
        args=parser.parse_args()
        ## main
        # read the .xml-file
        zip_ref = zipfile.ZipFile(args.azp_path)
        zip_ref.extractall('/tmp')
        tree = ET.parse('/tmp/content.xml')
        root = tree.getroot().findall('./{http://experience.univ-lyon1.fr/advene/ns}annotations')
        i=0
        end=[]
        begin=[]
        for child in root[0].iter():
            if child.get('type')=='#Shot':
                i+=1
                for child2 in child:
                    if child2.tag=='{http://experience.univ-lyon1.fr/advene/ns}millisecond-fragment':
                                   end.append(round(int(child2.get('end'))/1000*25))
                                   begin.append(round(int(child2.get('begin'))/1000*25))
                if i==2:
                    break
        vid_full=[]
        for start,stop in zip(begin,end):
            aux=read_video_segments('/home/jacob/Downloads/Wells_John_CompanyMen_full.mp4',start,stop,1080)
            aux = [x.astype('int') for x in aux]
            vid_full.append(aux)
        for j,frames in enumerate(vid_full):
            for i,frame in enumerate(frames):
                direc='/home/jacob/Downloads/vid'+str(j)+'_full'
                if not os.path.exists(direc):
                    os.makedirs(direc)
                name =direc+'/begin_'+str(i)+'.png'
                cv2.imwrite(name,frame)
        print('done')
