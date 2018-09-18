#####################################################
## libraries
#####################################################
import zipfile
import xml.etree.ElementTree as ET
import extract_dominant_color as edc
import argparse
import numpy as np
import cv2
######################################################
if __name__ == "__main__":
        ##############################################
        parser = argparse.ArgumentParser()
        parser.add_argument("azp_path",help="the path to a azp-file, a list of .azp-files or the path to a directory cotaining .azp-files")
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
        vid1=edc.read_video_segments('/home/jacob/Downloads/Wells_John_CompanyMen_full.mp4',end[0]-2,end[0])
        #vid2=edc.read_video_segments('/data/Wells_John_The_Company_Men.mp4',begin[1],begin[1]+2)
        print(len(vid1))
        cv2.imshow('s',vid1[0])
        #print(np.shape(vid1))
        print('#######################################')
        #print(vid2[0])
        print('done')
