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
    controller.register_importer(HPIImporter)
    return True
################################################################################################################
class HPIImporter(GenericImporter):
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
#        self.resolution_width=224
        self.image_scale=224  
        self.bin_threshold=5.0
        self.colors_to_return=5
        #################################
        self.model = "standard"
        self.confidence = 0.0
        self.detected_position = True

        self.create_relations = False
        self.split_types = False

        self.available_models = OrderedDict()
        self.available_models["standard"] = { 'id': "standard",
                                               'label': "Standard",
                                               'image_size': self.image_scale }

        self.optionparser.add_option(
            "-t", "--source-type-id", action="store", type="choice", dest="source_type_id",
            choices=[at.id for at in self.controller.package.annotationTypes],
            default=self.source_type_id,
            help=_("Type of annotation to analyze"),
            )
        self.optionparser.add_option(
            "-d", "--colorspace", action="store", type="choice", dest="colorspace",
            choices=['rgb','cie-lab'],
            default='rgb',
            help=_("defines the colorspace that is used by color extractor"),
            )

 #       self.optionparser.add_option(
 #           "-w", "--resolution_width", action="store", type="int",
 #           dest="resolution_width", default=self.resolution_width,
 #           help=_("set the scale with which the frames are extracted, default 200"),
 #           )
        self.optionparser.add_option(
            "-w", "--image_scale", action="store", type="int",
            dest="image_scale", default=self.image_scale,
            help=_("set the scale with which the frames are extracted, default 224"),
            )
        self.optionparser.add_option(
            "-b", "--bin_threshold", action="store", type="float",
            dest="bin_threshold", default=self.bin_threshold,
            help=_("set the percentage (0-100) a color has to reach to be returned,default 5"),
            )
        self.optionparser.add_option(
            "-c", "--colors_to_return", action="store", type="int",
            dest="colors_to_return", default=self.colors_to_return,
            help=_("set how many colors should be returned at maximum,default 5"),
            )
        self.optionparser.add_option(
            "-p", "--position", action="store_true",
            dest="detected_position", default=self.detected_position,
            help=_("Use detected position for created annotations"),
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
        print('unmet_requirements')
        """Check if external requirements for the importers are met.

        It returns a list of strings describing the unmet
        requirements. If the list is empty, then all requirements are
        met.
        """
        unmet_requirements = []
        # Make sure that we have all appropriate screenshots
        missing_screenshots = set()
        time_len=0
        for anno in self.source_type.annotations:
            #create timestamps per annotation 
#            for timestamp in range(anno.fragment.begin,anno.fragment.end): ## muss exakt gleich zu zeile 377 sein
#                if timestamp%16384==0:
#                   if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
#                      missing_screenshots.add(timestamp)
#                   time_len+=1
            for timestamp in (anno.fragment.begin,
                      int((anno.fragment.begin + anno.fragment.end) / 2),
                      anno.fragment.end):
                if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                    missing_screenshots.add(timestamp)
                time_len+=1
        if len(missing_screenshots) > 0:
            unmet_requirements.append(_("%d / %d screenshots are missing. Wait for extraction to complete.") % (len(missing_screenshots),3*len(self.source_type.annotations)))#time_len))
        return unmet_requirements
################################################################################################################
    def iterator(self):
        """I iterate over the created annotations.
        """
        print('image_scale: ',self.image_scale)
        print('colors_to_return: ', self.colors_to_return)
        print('bin_threshold: ', self.bin_threshold)
        self.progress(.1, "Sending request to server")
        if self.split_types:
            # Dict indexed by entity type name
            new_atypes = {}
        else:
            new_atype = self.ensure_new_type(
                "concept_%s" % self.source_type_id,
                title = _("Concepts for %s" % (self.source_type_id)))
            new_atype.mimetype = 'application/json'
            new_atype.setMetaData(config.data.namespace, "representation",
                                  'here/content/parsed/label')
        if self.create_relations:
            schema = self.create_schema('s_concept')
            rtype_id = 'concept_relation'
            rtype = self.package.get_element_by_id(rtype_id)
            if not rtype:
                # Create a relation type if it does not exist.
                rtype = schema.createRelationType(ident=rtype_id)
                rtype.author = config.data.get_userid()
                rtype.date = self.timestamp
                rtype.title = "Related concept"
                rtype.mimetype='text/plain'
                rtype.setHackedMemberTypes( ('*', '*') )
                schema.relationTypes.append(rtype)
                self.update_statistics('relation-type')
            if not hasattr(rtype, 'getHackedMemberTypes'):
                logger.error("%s is not a valid relation type" % rtype_id)
        
        image_scale = self.image_scale#self.available_models.get(self.model, {}).get('image_size')

        ori_list=[]
        im_bytesIO_ori=[]
        im_bytesIO_resi=[]
        buf_list=[]
        scaled_list=[]
        def get_scaled_image(t):
            """Return the image at the appropriate scale for the selected model.
            """
            print('timestamp',t)
            original = bytes(self.controller.package.imagecache.get(t))
            im = Image.open(BytesIO(original))
            im.save('/tmp/{0}.png'.format(t), 'PNG')
            #image_scale = False
            if image_scale:
                im = im.resize((image_scale, image_scale))
            pixbuf = np.asarray(im, dtype='uint8')[:,:,:3]
            return pixbuf
#        def get_scaled_image(t):
#            """Return the image at the appropriate scale for the selected model.
#            """
#            #print('timestamp',t)
#            original = bytes(self.controller.package.imagecache.get(t))
#            #print('ori_tyep: ', type(original))
#            ori_list.append(original)
#            #image_scale = False
#            if image_scale:
#                im = Image.open(BytesIO(original))
##                im_bytesIO_ori.append(im)

#                pix=np.asarray(im)
#                print([np.mean(pix[:,:,i]) for i in range(0,3)])
#                print('pix_shape: ',pix.shape)

#                im = im.resize((image_scale, image_scale))
#                scaled=np.asarray(im,dtype='uint8')[:,:,:3]
##                im_bytesIO_resi.append(im)

# #               buf = BytesIO()
##                buf_list.append(buf)
##                im.save(buf, 'PNG')#

##                scaled = buf.getvalue()
##                scaled_list.append(scaled)
##                buf.close()
#                print('scaled: ',np.shape(scaled))
#            else:
#                scaled = original
#            return scaled

        def fn_rgb_to_color(target_colorspace,path):
            if (path != 'full'):
                colors = {}
                with open(path) as f:
                    for line in f:
                        #split lines at "::
                        color, rgb = line.strip().split(':')
                        #strip the rgb-string of the parenthesis, split it up a the commas,
                        #cast them to int and put them into a tuples
                        rgb_value=tuple(map(int,(rgb.strip('(').strip(')').split(','))))
                        colors[color]=rgb_value
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
                print('HSV')
                for color in colors:
                    a = np.array((colors[color]),dtype='uint8')
                    b = a.reshape(1,1,3)
                    c = cv2.cvtColor(b,cv2.COLOR_RGB2HSV)
                    colors_aux[color]=tuple(c.reshape(3))
                colors=colors_aux
            if target_colorspace=='cie-lab':
                print('cie-lab')
                for color in colors:
                    a = np.array((colors[color]),dtype='uint8')
                    b = a.reshape(1,1,3)
                    c = cv2.cvtColor(b,cv2.COLOR_RGB2LAB)
                    colors_aux[color]=tuple(c.reshape(3))
                colors=colors_aux

            rgb_to_color={}
            for color in colors:
                rgb_to_color[colors[color]]=color
            #purple4 is median purple
            #skin is caucasian        
            return rgb_to_color

#fixme: use the option parser for bin_threshold, not the hardcoded value
        def extract_dominant_colors(frame_list,target_colorspace,path):
            print(str(len(frame_list))+' frames to process.')
            print('frame_list_shape:', np.mean(frame_list[0]) )
            rgb_to_color=fn_rgb_to_color(target_colorspace,path) #get the color dict 
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
            norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1] #normalize the binsi
            bins_norm={k:v/norm_factor for k,v in bins.items()}
            
            print('bins_norm: ',bins_norm)
 
            bin_threshold = 0
            colors_to_return = 10
            bins_sorted = sorted(list(zip(list(bins_norm.values()),list(bins_norm.keys()))),reverse=True)
            bins_sieved_dict={}
            for value,color in bins_sorted:
                if value > self.bin_threshold/100:
                    bins_sieved_dict[color]=value
            try:
                print('self.colors_to_return: ',self.colors_to_return)
                return bins_sieved_dict[:self.colors_to_return]
            except:
                return bins_sieved_dict

        response = {
            "model": 'self.model',
            'media_uri': 'self.package.uri',
            'media_filename': self.controller.get_default_media(),
            'annotations': []}

        for anno in self.source_type.annotations:
            frame_list=[]
#            for timestamp in range(anno.fragment.begin,anno.fragment.end):
#                if timestamp%16384==0:
#                   frame_list.append(get_scaled_image(timestamp))
            for timestamp in (anno.fragment.begin,
                      int((anno.fragment.begin + anno.fragment.end) / 2),
                      anno.fragment.end):
                frame_list.append(get_scaled_image(timestamp))
            print('frame_list_shape: ', np.shape(frame_list))
            annotations = { 'annotationid': anno.id,
                            'begin': anno.fragment.begin,
                            'end': anno.fragment.end,
                            'dominant_colors': extract_dominant_colors(frame_list,'rgb','full')}
            response['annotations'].append(annotations)

  #      print('ori_list: ',np.shape(ori_list[0]))
  #      print('im_bytesIO_ori',np.shape(im_bytesIO_ori[0])) 
  #      print('im_bytesIO_resi:',np.shape(im_bytesIO_resi[0]))
  #      print('buf_list: ',np.shape(buf_list[0]))
  #      print('scaled_list: ',np.shape(scaled_list[0]))

   #     stuff1=im_bytesIO_ori[0]
   #     for stuff in im_bytesIO_ori:
   #         assert stuff==stuff1

        output = json.dumps(response)
        progress = .2
        step = .8 / (len(output) or 1)
        self.progress(.2, _("Parsing %d results") % len(output))

        for anno in self.source_type.annotations:
            a = self.package.get_element_by_id(annotations['annotationid'])

            an = yield {
                'type': new_atype,
                'begin': anno.fragment.begin,
                'end': anno.fragment.end,
                'content': json.dumps(annotations['dominant_colors'])}

            if an is not None and self.create_relations:
                r = self.package.createRelation(
                    ident='_'.join( ('r', a.id, an.id) ),
                    type=rtype,
                    author=config.data.get_userid(),
                    date=self.timestamp,
                    members=(a, an))
                r.title = "Relation between %s and %s" % (a.id, an.id)
                self.package.relations.append(r)
                self.update_statistics('relation')
            self.progress(progress)
            progress += step
        print('太好了','\n',output)