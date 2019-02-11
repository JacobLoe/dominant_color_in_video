# Advene: Annotate Digital Videos, Exchange on the NEt
# Copyright (C) 2017 Olivier Aubert <contact@olivieraubert.net>
#
# Advene is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# Advene is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Advene; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
#

name="HPI dominant color extraction"
################################
#dominant_color libraries
################################
import cv2
import numpy as np
from sklearn.neighbors import KDTree
################################
import logging
logger = logging.getLogger(__name__)

from gettext import gettext as _

import base64
from collections import OrderedDict
from io import BytesIO
import json
from PIL import Image
import requests

import advene.core.config as config
import advene.util.helper as helper
from advene.util.importer import GenericImporter
################################################################################################################
def register(controller=None):
    controller.register_importer(HPIDCImporter)
    return True
################################################################################################################
class HPIDCImporter(GenericImporter):
    name = _("HPI dominant color extraction")
    annotation_filter = True
################################################################################################################
    def can_handle(fname):
        """Return a score between 0 and 100.

        100 is for the best match (specific extension), 0 is for no match at all.
        """
        return 80
    can_handle=staticmethod(can_handle)
################################################################################################################
    def __init__(self, author=None, package=None, defaulttype=None,
                 controller=None, callback=None, source_type=None):
        GenericImporter.__init__(self,
                                 author=author,
                                 package=package,
                                 defaulttype=defaulttype,
                                 controller=controller,
                                 callback=callback,
                                 source_type=source_type)
        if self.source_type is None:
            self.source_type = self.controller.package.annotationTypes[0]
        self.source_type_id = self.source_type.id

        if source_type is not None:
            # A source_type was specified at instanciation. Update the
            # preferences now since we will use this info to update
            # the filter options.
            self.get_preferences().update({'source_type_id': self.source_type_id})
        ##################################
        self.image_scale=224
        self.min_bin_threshold=5.0
        self.max_bin_threshold=60.0
        self.colorspace='cie-lab'
        self.colors_used='darkred,firebrick,crimson,red,tomato,salmon,darkorange,gold,darkkhaki,yellow,darkolivegreen,olivedrab,greenyellow,darkgreen,aquamarine,steelblue,skyblue,darkblue,blue,royalblue,purple,violet,deeppink,pink,antiquewhite,saddlebrown,sandybrown,ivory,dimgrey,grey,silver,lightgrey,black,white,darkcyan,cyan,green,khaki,goldenrod,orange,coral,magenta,wheat,skin,purple4'
        self.image_timestamp_divider=1000#16384 #16384 results in roughly 30 images per annotation
        #################################

        self.optionparser.add_option(
            "-t", "--source-type-id", action="store", type="choice", dest="source_type_id",
            choices=[at.id for at in self.controller.package.annotationTypes],
            default=self.source_type_id,
            help=_("Type of annotation to analyze"),
            )
        self.optionparser.add_option(
            "-d", "--colorspace", action="store", type="choice", dest="colorspace",
            choices=['rgb','cie-lab'],
            default=self.colorspace,
            help=_("defines the colorspace that is used by color extractor"),
            )
        self.optionparser.add_option(
            "-b", "--min_color_threshold", action="store", type="float",
            dest="min_bin_threshold", default=self.min_bin_threshold,
            help=_("sets the minimum percentage (0-100) a color has to reach to be returned,default 5.0"),
            )
        self.optionparser.add_option(
            "-c", "--max_color_threshold", action="store", type="float",
            dest="max_bin_threshold", default=self.max_bin_threshold,
            help=_("sets the maximum percentage (0-100) a color can reach before it is not returned,default 60.0"),
            )
        self.optionparser.add_option(
            "-e","--colors_used",action="store",type="string",
            dest="colors_used",
            help=_("defines the colors that are used for the color extraction, format is color1,color2,color3 ,colors have to be on the full list, the default value is a list of 40 colors"),
            )
################################################################################################################
    @staticmethod
    def can_handle(fname):
        """
        """
        if True:#'http' in fname:
            return 100
        else:
            return 0
################################################################################################################
    def process_file(self, _filename):
        self.convert(self.iterator())
################################################################################################################
    def check_requirements(self):
        """Check if external requirements for the importers are met.

        It returns a list of strings describing the unmet
        requirements. If the list is empty, then all requirements are
        met.
        """
        unmet_requirements = []
        ########################################################################
        if self.min_bin_threshold>100 or self.max_bin_threshold>100:
           unmet_requirements.append(_("color thresholds can't be more than 100"))
        if self.min_bin_threshold<0 or self.max_bin_threshold<0:
           unmet_requirements.append(_("color thresholds can't be negative"))
        if self.min_bin_threshold>self.max_bin_threshold:
           unmet_requirements.append(_("max_color_threshold must be higher than min_color_threshold"))
        #######################################################################
        colors_reference={'darkred':(139,0,0),'firebrick':(178,34,34),'crimson':(220,20,60),'red':(255,0,0),
                    'tomato':(255,99,71),'salmon':(250,128,114),'darkorange':(255,140,0),'gold':(255,215,0),
                    'darkkhaki':(189,183,107),'yellow':(255,255,0),'darkolivegreen':(85,107,47),'olivedrab':(107,142,35),
                    'greenyellow':(173,255,47),'darkgreen':(0,100,0),'aquamarine':(127,255,212),'steelblue':(70,130,180),
                    'skyblue':(135,206,235),'darkblue':(0,0,139),'blue':(0,0,255),'royalblue':(65,105,225),'purple':(128,0,128),
                    'violet':(238,130,238),'deeppink':(255,20,147),'pink':(255,192,203),'antiquewhite':(250,235,215),
                    'saddlebrown':(139,69,19),'sandybrown':(244,164,96),'ivory':(255,255,240),'dimgrey':(105,105,105),
                    'grey':(28,128,128),'silver':(192,192,192),'lightgrey':(211,211,211),'black':(0,0,0),'white':(255,255,255),
                    'darkcyan':(0,139,139),'cyan':(0,255,255),'green':(0,128,0),'khaki':(240,230,140),'goldenrod':(218,165,32),
                    'orange':(255,165,0),'coral':(255,127,80),'magenta':(255,0,255),'wheat':(245,222,179),'skin':(255,224,189),'purple4':(147,112,219)}
        colors_used=self.colors_used
        wrong_colors=[]
        if (colors_used!=''):
            colors={}

            colors_used_aux1=colors_used.split(',')
            colors_used_aux2=colors_used.split(';')
            if len(colors_used_aux1)>len(colors_used_aux2) and len(colors_used_aux2)==1:
                colors_used=colors_used_aux1
            elif len(colors_used_aux1)<len(colors_used_aux2) and len(colors_used_aux1)==1:
                colors_used=colors_used_aux2
            elif len(colors_used_aux1)==1 and len(colors_used_aux2)==1:
                colors_used=[colors_used]
            for color in colors_used:
                try:
                    colors[color]=colors_reference[color]
                except:
                    wrong_colors.append(color)
        
        if len(wrong_colors) > 0:
            unmet_requirements.append(_("There are errors in the color list. The wrong colors are: %d") %str(wrong_colors))
        #######################################################################
        # Make sure that we have all appropriate screenshots
        missing_screenshots = set()
        time_len=0
        for anno in self.source_type.annotations:
            #create timestamps per annotation 
            for i,timestamp in enumerate(range(anno.fragment.begin,anno.fragment.end)):
                if i==0:
                   if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                      missing_screenshots.add(timestamp)
                elif timestamp%self.image_timestamp_divider==0 or timestamp==int((anno.fragment.begin + anno.fragment.end) / 2):
                   if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                      missing_screenshots.add(timestamp)
                   time_len+=1
                elif i==len(range(anno.fragment.begin,anno.fragment.end)):
                   if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                      missing_screenshots.add(timestamp)
        if len(missing_screenshots) > 0:
            unmet_requirements.append(_("%d / %d screenshots are missing. Wait for extraction to complete.") % (len(missing_screenshots),time_len*len(self.source_type.annotations)))
        return unmet_requirements
################################################################################################################
    def iterator(self):
        """I iterate over the created annotations.
        """
        self.progress(.1, "Sending request to server")
        self.source_type = self.controller.package.get_element_by_id(self.source_type_id)
        new_atype = self.ensure_new_type(
                "concept_%s" % self.source_type_id,
                title = _("Concepts for %s" % (self.source_type_id)))
        new_atype.mimetype = 'application/json'
        new_atype.setMetaData(config.data.namespace, "representation",'here/content/parsed/label')
        
############################
# get the images from cache
############################
        def get_scaled_image(t):
            """Return the image at the appropriate scale for the selected model.
            """
            #print('timestamp: ',t)
            original = bytes(self.controller.package.imagecache.get(t))
            im = Image.open(BytesIO(original))
            im.save('/tmp/{0}.png'.format(t), 'PNG')
            if self.image_scale:
                im = im.resize((self.image_scale, self.image_scale))
            pixbuf = np.asarray(im, dtype='uint8')[:,:,:3]
            return pixbuf

############################
# get the colors used for the color extraction
#############################
        def fn_rgb_to_color(target_colorspace,colors_used):
            # rgb-values taken from: https://www.rapidtables.com/web/color/RGB_Color.html
            #purple4 is taken as median purple
            #skin is taken as caucasian  
            colors_reference={'darkred':(139,0,0),'firebrick':(178,34,34),'crimson':(220,20,60),'red':(255,0,0),
                        'tomato':(255,99,71),'salmon':(250,128,114),'darkorange':(255,140,0),'gold':(255,215,0),
                        'darkkhaki':(189,183,107),'yellow':(255,255,0),'darkolivegreen':(85,107,47),'olivedrab':(107,142,35),
                        'greenyellow':(173,255,47),'darkgreen':(0,100,0),'aquamarine':(127,255,212),'steelblue':(70,130,180),
                        'skyblue':(135,206,235),'darkblue':(0,0,139),'blue':(0,0,255),'royalblue':(65,105,225),'purple':(128,0,128),
                        'violet':(238,130,238),'deeppink':(255,20,147),'pink':(255,192,203),'antiquewhite':(250,235,215),
                        'saddlebrown':(139,69,19),'sandybrown':(244,164,96),'ivory':(255,255,240),'dimgrey':(105,105,105),
                        'grey':(28,128,128),'silver':(192,192,192),'lightgrey':(211,211,211),'black':(0,0,0),'white':(255,255,255),
                        'darkcyan':(0,139,139),'cyan':(0,255,255),'green':(0,128,0),'khaki':(240,230,140),'goldenrod':(218,165,32),
                        'orange':(255,165,0),'coral':(255,127,80),'magenta':(255,0,255),'wheat':(245,222,179),'skin':(255,224,189),'purple4':(147,112,219)}
            if (colors_used!=''):
                colors={}

                colors_used_aux1=colors_used.split(',')
                colors_used_aux2=colors_used.split(';')
                if len(colors_used_aux1)>len(colors_used_aux2):
                    colors_used=colors_used_aux1
                else:
                    colors_used=colors_used_aux2

                wrong_colors=[]
                for color in colors_used:
                    try:
                        colors[color]=colors_reference[color]
                    except:
                        wrong_colors.append(color)
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
            if target_colorspace=='HSV':
                for color in colors:
                    a = np.array((colors[color]),dtype='uint8')
                    b = a.reshape(1,1,3)
                    c = cv2.cvtColor(b,cv2.COLOR_RGB2HSV)
                    colors_aux[color]=tuple(c.reshape(3))
                colors=colors_aux
            if target_colorspace=='cie-lab':
                for color in colors:
                    a = np.array((colors[color]),dtype='uint8')
                    b = a.reshape(1,1,3)
                    c = cv2.cvtColor(b,cv2.COLOR_RGB2LAB)
                    colors_aux[color]=tuple(c.reshape(3))
                colors=colors_aux

            rgb_to_color={}
            for color in colors:
                rgb_to_color[colors[color]]=color
      
            return rgb_to_color
#################################
# dominant color extractor
################################
        def extract_dominant_colors(frame_list,target_colorspace,colors_used):
            print('extract_dominant_colors')
            rgb_to_color=fn_rgb_to_color(target_colorspace,colors_used) #get the color dict 
            bins={} #bins dict for histograms 
            for rgb in rgb_to_color: #init the dict with zeros for every key
                bins[rgb_to_color[rgb]]=0
            rgb_list=[] #create a traverseable list of the rgb_values
            for rgb in rgb_to_color: #map the values of the dict to a list
                rgb_list.append(rgb)
            kdt = KDTree(rgb_list, leaf_size=30, metric='euclidean')

            for image in frame_list: #traverse the given fragment
                img = image.reshape((image.shape[0] * image.shape[1], 3)) #flatten the image to 1d   
                nns = kdt.query(img, k=1, return_distance=False)
                for nn in nns:
                    bins[rgb_to_color[rgb_list[nn[0]]]]+=1

            #print('np.shape(frame_list): ',np.shape(frame_list))
            norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1] #normalize the bins
            bins_norm={k:v/norm_factor for k,v in bins.items()}

            bins_sorted = sorted(list(zip(list(bins_norm.values()),list(bins_norm.keys()))),reverse=True)
            bins_sieved_dict={}
            for value,color in bins_sorted:
                if value >= self.min_bin_threshold/100 and value <= self.max_bin_threshold/100:
                    bins_sieved_dict[color]=value*100
            keys = list(bins_sieved_dict.keys())
            return keys#bins_sieved_dict

######################################
# extract the dominant colors
######################################

        response = {
            "model": 'standard',
            'media_uri': 'self.package.uri',
            'media_filename': self.controller.get_default_media(),
            'annotations': []}

        #print('annotate')
        for anno in self.source_type.annotations:
            frame_list = []
            #print('anno: ',anno)
            #print('anno.fragment.begin: ',anno.fragment.begin)
            #print('anno.fragment.end: ',anno.fragment.end)
            for i,timestamp in enumerate(range(anno.fragment.begin,anno.fragment.end)):
                if i==0:
                   frame_list.append(get_scaled_image(timestamp))
                #print('modulo: ',timestamp%self.image_timestamp_divider==0)
                elif timestamp%self.image_timestamp_divider==0 or timestamp==int((anno.fragment.begin + anno.fragment.end) / 2):
                   #print('timestamp: ',timestamp)
                   frame_list.append(get_scaled_image(timestamp))
                elif i==len(range(anno.fragment.begin,anno.fragment.end)):
                   frame_list.append(get_scaled_image(timestamp))
            annotations = { 'annotationid': anno.id,
                            'begin': anno.fragment.begin,
                            'end': anno.fragment.end,
                            'dominant_colors': extract_dominant_colors(frame_list,self.colorspace,self.colors_used)}
            response['annotations'].append(annotations)

######################################
# create histogram of extracted colors
#########################################

        bins={} #bins dict for histograms
        rgb_to_color=fn_rgb_to_color(self.colorspace,self.colors_used) #get the color dict 
        for rgb in rgb_to_color: #init the histogram-dict with zeros for every key
            bins[rgb_to_color[rgb]]=0
        for anno in response['annotations']:
            for color in anno['dominant_colors']:
                bins[color]+=1
        #print('bins: ',bins)

################################################
# write the dominant colors into advene gui
################################################
        output = json.dumps(response)

        progress = .2
        step = .8 / (len(output) or 1)
        self.progress(.2, _("Parsing %d results") % len(output))

        for anno in response['annotations']:
            a = self.package.get_element_by_id(anno['annotationid'])
            an = yield {
                'type': new_atype,
                'begin': anno['begin'],
                'end': anno['end'],
                'content': json.dumps(anno['dominant_colors'])}
            self.progress(progress)
            progress += step
