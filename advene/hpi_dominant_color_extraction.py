#
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
        self.resolution_width=0
        self.bin_threshold=0.0
        self.colors_to_return=0
        #################################
        self.model = "standard"
        self.confidence = 0.0
        self.detected_position = True
        self.split_types = False
        #self.url = self.get_preferences().get('url', 'http://localhost:9000/')

        self.server_options = {}


        # Populate available models options from server
        #try:
        #    r = requests.get(self.url)
        #    if r.status_code == 200:
        #        # OK. We should have some server options available as json
        #        data = r.json()
        #        caps = data.get('data', {}).get('capabilities', {})
        #        for n in ('minimum_batch_size', 'maximum_batch_size', 'available_models'):
        #            self.server_options[n] = caps.get(n, None)
        #        logger.warn("Got capabilities from VCD server - batch size in (%d, %d) - %d models: %s",
        #                    self.server_options['minimum_batch_size'],
        #                    self.server_options['maximum_batch_size'],
        #                    len(self.server_options['available_models']),
        #                    ", ".join(item['id'] for item in self.server_options['available_models']))
        #except requests.exceptions.RequestException:
        #    pass


        #if 'available_models' in self.server_options:
        #    self.available_models = OrderedDict((item['id'], item) for item in self.server_options['available_models'])
        #else:
        #    self.available_models = OrderedDict()
        #    self.available_models["standard"] = { 'id': "standard",
        #                                          'label': "Standard",
        #                                          'image_size': 224 }

        self.optionparser.add_option(
            "-t", "--source-type-id", action="store", type="choice", dest="source_type_id",
            choices=[at.id for at in self.controller.package.annotationTypes],
            default=self.source_type_id,
            help=_("Type of annotation to analyze"),
            )
        self.optionparser.add_option(
            "-w", "--resolution_width", action="store", type="int",
            dest="resolution_width", default=200,
            help=_("set the resolution width of the videofile, the resolution scales automatically to 16:9"),
            )
        self.optionparser.add_option(
            "-b", "--bin_threshold", action="store", type="float",
            dest="bin_threshold", default=5.0,
            help=_("set the percentage (0-100) a color has to reach to be returned,default 5"),
            )
        self.optionparser.add_option(
            "-c", "--colors_to_return", action="store", type="int",
            dest="colors_to_return", default=5,
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
        if 'http' in fname:
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

        # Check server connectivity
        #try:
        #    requests.get(self.url)
        #except requests.exceptions.RequestException:
        #    unmet_requirements.append(_("Cannot connect to VCD server. Check that it is running and accessible."))

        # Make sure that we have all appropriate screenshots
       # missing_screenshots = set()
       # for a in self.source_type.annotations:
       #     for t in (a.fragment.begin,
       #               int((a.fragment.begin + a.fragment.end) / 2),
       #               a.fragment.end):
       #         if self.controller.get_snapshot(annotation=a, position=t).is_default:
       #             missing_screenshots.add(t)
       # if len(missing_screenshots) > 0:
       #     unmet_requirements.append(_("%d / %d screenshots are missing. Wait for extraction to complete.") % (len(missing_screenshots),
       #                                                                                                         3 * len(self.source_type.annotations)))
        return unmet_requirements
################################################################################################################
    def iterator(self):
        """I iterate over the created annotations.
        """
        #logger.warn("Importing using %s model", self.model)
        self.source_type = self.controller.package.get_element_by_id(self.source_type_id)

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
        ################################################################################################################
        def extract_dominant_colors(frame_list,target_colorspace,path,what_to_process):
            print(str(len(frame_list))+' frames to process.')
            rgb_to_color=fn_rgb_to_color(target_colorspace,path) #get the color dict 
            bins={} #bins dict for histograms 
            for rgb in rgb_to_color: #init the dict with zeros for every key
                bins[rgb_to_color[rgb]]=0
            rgb_list=[] #create a traverseable list of the rgb_values
            for rgb in rgb_to_color: #map the values of the dict to a list
                rgb_list.append(rgb)

            kdt = KDTree(rgb_list, leaf_size=30, metric='euclidean')
            if what_to_process=='scene':
               for image in frame_list: #traverse the video
                   img = image.reshape((image.shape[0] * image.shape[1], 3)) #flatten the image to 1d   
                   nns = kdt.query(img, k=1, return_distance=False)
                   for nn in nns:
                       bins[rgb_to_color[rgb_list[nn[0]]]]+=1
               norm_factor = len(frame_list)* np.shape(frame_list[0])[0] * np.shape(frame_list[0])[1] #normalize the bins
               bins_norm={k:v/norm_factor for k,v in bins.items()}
            return bins_norm
        ################################################################################################################
        # Use a requests.session to use a KeepAlive connection to the server
        session = requests.session()

        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        response = session.post(headers=headers, json={
            "model": 'self.model',
            'media_uri': 'self.package.uri',
            'media_filename': self.controller.get_default_media(),
            'annotations': [
                { 'annotationid': a.id,
                  'begin': a.fragment.begin,
                  'end': a.fragment.end,
                  'dominant_colors': ['Test','test']
                }
                for a in self.source_type.annotations
            ]
        })


        output = response.json()
        if output.get('status') != 200:
            # Not OK result. Display error message.
            msg = _("Server error: %s") % output.get('message', _("Server transmission error."))
            logger.error(msg)
            self.output_message = msg
            return
        # FIXME: maybe check consistency with media_filename/media_uri?
        concepts = output.get('data', {}).get('concepts', [])
        progress = .2
        step = .8 / (len(concepts) or 1)
        self.progress(.2, _("Parsing %d results") % len(concepts))
        logger.warn(_("Parsing %d results (level %f)") % (len(concepts), self.confidence))
        for item in concepts:
            print(item)
            dominant_colors=0
            an = yield {
                'type': new_atype,
                'begin': begin,
                'end': end,
                'content': json.dumps(dominant_colors),
            }
            a = self.package.get_element_by_id(item['annotationid'])
            if self.detected_position:
                begin = item['timecode']
            else:
                begin = a.fragment.begin
            end = a.fragment.end
            label = item.get('label')
            label_id = helper.title2id(label)
            if label and self.split_types:
                new_atype = new_atypes.get(label_id)
                if new_atype is None:
                   # Not defined yet. Create a new one.
                   new_atype = self.ensure_new_type(label_id, title = _("%s concept" % label))
                   new_atype.mimetype = 'application/json'
                   new_atype.setMetaData(config.data.namespace, "representation",
                                         'here/content/parsed/label')
                   new_atypes[label_id] = new_atype
            an = yield {
                'type': new_atype,
                'begin': begin,
                'end': end,
                'content': json.dumps(item),
            }
            self.progress(progress)
            progress += step
################################################################################################################
