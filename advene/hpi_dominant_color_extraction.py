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

import logging

from gettext import gettext as _

from io import BytesIO
import json
import PIL
import math

import advene.core.config as config
from advene.util.importer import GenericImporter

import cv2
import numpy as np

import scipy.spatial
from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.neighbors import KDTree

name="HPI dominant color extraction"
logger = logging.getLogger(__name__)

# rgb-values taken from: https://www.rapidtables.com/web/color/RGB_Color.html
# purple4 is taken as median purple
# skin is taken as caucasian
__SUPPORTED_COLORS_RGB__ = {'darkred': (139, 0, 0), 'firebrick': (178, 34, 34), 'crimson': (220, 20, 60),
                            'red': (255, 0, 0), 'tomato': (255, 99, 71), 'salmon': (250, 128, 114),
                            'darkorange': (255, 140, 0), 'gold': (255, 215, 0), 'darkkhaki': (189, 183, 107),
                            'yellow': (255, 255, 0), 'darkolivegreen': (85, 107, 47), 'olivedrab': (107, 142, 35),
                            'greenyellow': (173, 255, 47), 'darkgreen': (0, 100, 0), 'aquamarine': (127, 255, 212),
                            'steelblue': (70, 130, 180), 'skyblue': (135, 206, 235), 'darkblue': (0, 0, 139),
                            'blue': (0, 0, 255), 'royalblue': (65, 105, 225), 'purple': (128, 0, 128),
                            'violet': (238, 130, 238), 'deeppink': (255, 20, 147), 'pink': (255, 192, 203),
                            'antiquewhite': (250, 235, 215), 'saddlebrown': (139, 69, 19), 'sandybrown': (244, 164, 96),
                            'ivory': (255, 255, 240), 'dimgrey': (105, 105, 105), 'grey': (28, 128, 128),
                            'silver': (192, 192, 192), 'lightgrey': (211, 211, 211), 'black': (0, 0, 0),
                            'white': (255, 255, 255), 'khaki': (240, 230, 140), 'goldenrod': (218, 165, 32),
                            'orange': (255, 165, 0), 'coral': (255, 127, 80), 'magenta': (255, 0, 255),
                            'wheat': (245, 222, 179), 'skin': (255, 224, 189), 'purple4': (147, 112, 219)}


def register(controller=None):
    controller.register_importer(HPIDCImporter)
    return True


class cached_property(object):
    """Decorator that creates converts a method with a single
    self argument into a property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type):
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res


class MMCQ(object):
    """Basic Python port of the MMCQ (modified median cut quantization)
    algorithm from the Leptonica library (http://www.leptonica.com/).
    """

    SIGBITS = 5
    RSHIFT = 8 - SIGBITS
    MAX_ITERATION = 1000
    FRACT_BY_POPULATIONS = 0.75

    @staticmethod
    def get_color_index(r, g, b):
        return (r << (2 * MMCQ.SIGBITS)) + (g << MMCQ.SIGBITS) + b

    @staticmethod
    def get_histo(pixels):
        """histo (1-d array, giving the number of pixels in each quantized
        region of color space)
        """
        histo = dict()
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            index = MMCQ.get_color_index(rval, gval, bval)
            histo[index] = histo.setdefault(index, 0) + 1
        return histo

    @staticmethod
    def vbox_from_pixels(pixels, histo):
        rmin = 1000000
        rmax = 0
        gmin = 1000000
        gmax = 0
        bmin = 1000000
        bmax = 0
        for pixel in pixels:
            rval = pixel[0] >> MMCQ.RSHIFT
            gval = pixel[1] >> MMCQ.RSHIFT
            bval = pixel[2] >> MMCQ.RSHIFT
            rmin = min(rval, rmin)
            rmax = max(rval, rmax)
            gmin = min(gval, gmin)
            gmax = max(gval, gmax)
            bmin = min(bval, bmin)
            bmax = max(bval, bmax)
        return VBox(rmin, rmax, gmin, gmax, bmin, bmax, histo)

    @staticmethod
    def median_cut_apply(histo, vbox):
        if not vbox.count:
            return (None, None)

        rw = vbox.r2 - vbox.r1 + 1
        gw = vbox.g2 - vbox.g1 + 1
        bw = vbox.b2 - vbox.b1 + 1
        maxw = max([rw, gw, bw])
        # only one pixel, no split
        if vbox.count == 1:
            return (vbox.copy, None)
        # Find the partial sum arrays along the selected axis.
        total = 0
        sum_ = 0
        partialsum = {}
        lookaheadsum = {}
        do_cut_color = None
        if maxw == rw:
            do_cut_color = 'r'
            for i in range(vbox.r1, vbox.r2+1):
                sum_ = 0
                for j in range(vbox.g1, vbox.g2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(i, j, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        elif maxw == gw:
            do_cut_color = 'g'
            for i in range(vbox.g1, vbox.g2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.b1, vbox.b2+1):
                        index = MMCQ.get_color_index(j, i, k)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        else:  # maxw == bw
            do_cut_color = 'b'
            for i in range(vbox.b1, vbox.b2+1):
                sum_ = 0
                for j in range(vbox.r1, vbox.r2+1):
                    for k in range(vbox.g1, vbox.g2+1):
                        index = MMCQ.get_color_index(j, k, i)
                        sum_ += histo.get(index, 0)
                total += sum_
                partialsum[i] = total
        for i, d in partialsum.items():
            lookaheadsum[i] = total - d

        # determine the cut planes
        dim1 = do_cut_color + '1'
        dim2 = do_cut_color + '2'
        dim1_val = getattr(vbox, dim1)
        dim2_val = getattr(vbox, dim2)
        for i in range(dim1_val, dim2_val+1):
            if partialsum[i] > (total / 2):
                vbox1 = vbox.copy
                vbox2 = vbox.copy
                left = i - dim1_val
                right = dim2_val - i
                if left <= right:
                    d2 = min([dim2_val - 1, int(i + right / 2)])
                else:
                    d2 = max([dim1_val, int(i - 1 - left / 2)])
                # avoid 0-count boxes
                while not partialsum.get(d2, False):
                    d2 += 1
                count2 = lookaheadsum.get(d2)
                while not count2 and partialsum.get(d2-1, False):
                    d2 -= 1
                    count2 = lookaheadsum.get(d2)
                # set dimensions
                setattr(vbox1, dim2, d2)
                setattr(vbox2, dim1, getattr(vbox1, dim2) + 1)
                return (vbox1, vbox2)
        return (None, None)

    @staticmethod
    def quantize(pixels, max_color):
        """Quantize.
        :param pixels: a list of pixel in the form (r, g, b)
        :param max_color: max number of colors
        """
        if not pixels:
            raise Exception('Empty pixels when quantize.')
        if max_color < 2 or max_color > 256:
            raise Exception('Wrong number of max colors when quantize.')

        histo = MMCQ.get_histo(pixels)

        # check that we aren't below maxcolors already
        if len(histo) <= max_color:
            # generate the new colors from the histo and return
            pass

        # get the beginning vbox from the colors
        vbox = MMCQ.vbox_from_pixels(pixels, histo)
        pq = PQueue(lambda x: x.count)
        pq.push(vbox)

        # inner function to do the iteration
        def iter_(lh, target):
            n_color = 1
            n_iter = 0
            while n_iter < MMCQ.MAX_ITERATION:
                vbox = lh.pop()
                if not vbox.count:  # just put it back
                    lh.push(vbox)
                    n_iter += 1
                    continue
                # do the cut
                vbox1, vbox2 = MMCQ.median_cut_apply(histo, vbox)
                if not vbox1:
                    raise Exception("vbox1 not defined; shouldn't happen!")
                lh.push(vbox1)
                if vbox2:  # vbox2 can be null
                    lh.push(vbox2)
                    n_color += 1
                if n_color >= target:
                    return
                if n_iter > MMCQ.MAX_ITERATION:
                    return
                n_iter += 1

        # first set of colors, sorted by population
        iter_(pq, MMCQ.FRACT_BY_POPULATIONS * max_color)

        # Re-sort by the product of pixel occupancy times the size in
        # color space.
        pq2 = PQueue(lambda x: x.count * x.volume)
        while pq.size():
            pq2.push(pq.pop())

        # next set - generate the median cuts using the (npix * vol) sorting.
        iter_(pq2, max_color - pq2.size())

        # calculate the actual colors
        cmap = CMap()
        while pq2.size():
            cmap.push(pq2.pop())
        return cmap


class VBox(object):
    """3d color space box"""
    def __init__(self, r1, r2, g1, g2, b1, b2, histo):
        self.r1 = r1
        self.r2 = r2
        self.g1 = g1
        self.g2 = g2
        self.b1 = b1
        self.b2 = b2
        self.histo = histo

    @cached_property
    def volume(self):
        sub_r = self.r2 - self.r1
        sub_g = self.g2 - self.g1
        sub_b = self.b2 - self.b1
        return (sub_r + 1) * (sub_g + 1) * (sub_b + 1)

    @property
    def copy(self):
        return VBox(self.r1, self.r2, self.g1, self.g2,
                    self.b1, self.b2, self.histo)

    @cached_property
    def avg(self):
        ntot = 0
        mult = 1 << (8 - MMCQ.SIGBITS)
        r_sum = 0
        g_sum = 0
        b_sum = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    histoindex = MMCQ.get_color_index(i, j, k)
                    hval = self.histo.get(histoindex, 0)
                    ntot += hval
                    r_sum += hval * (i + 0.5) * mult
                    g_sum += hval * (j + 0.5) * mult
                    b_sum += hval * (k + 0.5) * mult

        if ntot:
            r_avg = int(r_sum / ntot)
            g_avg = int(g_sum / ntot)
            b_avg = int(b_sum / ntot)
        else:
            r_avg = int(mult * (self.r1 + self.r2 + 1) / 2)
            g_avg = int(mult * (self.g1 + self.g2 + 1) / 2)
            b_avg = int(mult * (self.b1 + self.b2 + 1) / 2)

        return r_avg, g_avg, b_avg

    def contains(self, pixel):
        rval = pixel[0] >> MMCQ.RSHIFT
        gval = pixel[1] >> MMCQ.RSHIFT
        bval = pixel[2] >> MMCQ.RSHIFT
        return all([
            rval >= self.r1,
            rval <= self.r2,
            gval >= self.g1,
            gval <= self.g2,
            bval >= self.b1,
            bval <= self.b2,
        ])

    @cached_property
    def count(self):
        npix = 0
        for i in range(self.r1, self.r2 + 1):
            for j in range(self.g1, self.g2 + 1):
                for k in range(self.b1, self.b2 + 1):
                    index = MMCQ.get_color_index(i, j, k)
                    npix += self.histo.get(index, 0)
        return npix


class CMap(object):
    """Color map"""
    def __init__(self):
        self.vboxes = PQueue(lambda x: x['vbox'].count * x['vbox'].volume)

    @property
    def palette(self):
        return self.vboxes.map(lambda x: x['color'])

    def push(self, vbox):
        self.vboxes.push({
            'vbox': vbox,
            'color': vbox.avg,
        })

    def size(self):
        return self.vboxes.size()

    def nearest(self, color):
        d1 = None
        p_color = None
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            d2 = math.sqrt(
                math.pow(color[0] - vbox['color'][0], 2) +
                math.pow(color[1] - vbox['color'][1], 2) +
                math.pow(color[2] - vbox['color'][2], 2)
            )
            if d1 is None or d2 < d1:
                d1 = d2
                p_color = vbox['color']
        return p_color

    def map(self, color):
        for i in range(self.vboxes.size()):
            vbox = self.vboxes.peek(i)
            if vbox['vbox'].contains(color):
                return vbox['color']
        return self.nearest(color)


class PQueue(object):
    """Simple priority queue."""
    def __init__(self, sort_key):
        self.sort_key = sort_key
        self.contents = []
        self._sorted = False

    def sort(self):
        self.contents.sort(key=self.sort_key)
        self._sorted = True

    def push(self, o):
        self.contents.append(o)
        self._sorted = False

    def peek(self, index=None):
        if not self._sorted:
            self.sort()
        if index is None:
            index = len(self.contents) - 1
        return self.contents[index]

    def pop(self):
        if not self._sorted:
            self.sort()
        return self.contents.pop()

    def size(self):
        return len(self.contents)

    def map(self, f):
        return list(map(f, self.contents))


class HPIDCImporter(GenericImporter):
    name = _("HPI dominant color extraction")
    annotation_filter = True

    def can_handle(fname):
        """Return a score between 0 and 100.

        100 is for the best match (specific extension), 0 is for no match at all.
        """
        return 80
    can_handle = staticmethod(can_handle)

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
            # A source_type was specified at instantiation. Update the
            # preferences now since we will use this info to update
            # the filter options.
            self.get_preferences().update({'source_type_id': self.source_type_id})

        ##################################
        self.image_scale = 224            # FIXME: verify if using unscaled image is robust/fast enough
        #self.min_bin_threshold = 5.0
        #self.max_bin_threshold = 60.0
        self.colorspace = 'cie-lab'
        self.color_sel = ",".join(__SUPPORTED_COLORS_RGB__.keys())
        self.colors = dict()
        self.image_timestamp_divider = 1000  # 16384 #16384 results in roughly 30 images per annotation
        #################################

        self.optionparser.add_option(
            "-t", "--source-type-id", action="store", type="choice", dest="source_type_id",
            choices=[at.id for at in self.controller.package.annotationTypes],
            default=self.source_type_id,
            help=_("Type of annotation to analyze"),
            )
        self.optionparser.add_option(
            "-c", "--colorspace", action="store", type="choice", dest="colorspace",
            choices=['rgb', 'hsv', 'cie-lab'],
            default='cie-lab',
            help=_("defines the colorspace that is used by color extractor"),
            )
        #self.optionparser.add_option(
        #    "-b", "--min_color_threshold", action="store", type="float",
        #    dest="min_bin_threshold", default=5.0,
        #    help=_("sets the minimum percentage (0-100) a color has to reach to be returned,default 5.0"),
        #    )
        #self.optionparser.add_option(
        #    "-c", "--max_color_threshold", action="store", type="float",
        #    dest="max_bin_threshold", default=60.0,
        #    help=_("sets the maximum percentage (0-100) a color can reach before it is not returned,default 60.0"),
        #    )
        self.optionparser.add_option(
            "-e", "--color_sel", action="store", type="string",
            dest="color_sel", default=",".join(__SUPPORTED_COLORS_RGB__.keys()),
            help=_(
                "Defines the colors that are used for the color extraction. " +
                "Format is color1,color2,color3,... " +
                "Colors need to be selected from the list of supported colors (see default)."),
        )

    def process_file(self, _filename):
        self.convert(self.iterator())

    def check_requirements(self):
        """Check if external requirements for the importers are met.

        It returns a list of strings describing the unmet
        requirements. If the list is empty, then all requirements are
        met.
        """
        unmet_requirements = []

        #if self.min_bin_threshold > 100 or self.max_bin_threshold > 100:
        #    unmet_requirements.append(_("color thresholds can't be more than 100"))

        #if self.min_bin_threshold < 0 or self.max_bin_threshold < 0:
        #    unmet_requirements.append(_("color thresholds can't be negative"))

        #if self.min_bin_threshold > self.max_bin_threshold:
        #    unmet_requirements.append(_("max_color_threshold must be higher than min_color_threshold"))

        unk_colors = []
        for col in self.color_sel.split(','):
            if col not in __SUPPORTED_COLORS_RGB__:
                unk_colors.append(col)
            else:
                rgb = np.asarray(__SUPPORTED_COLORS_RGB__[col], dtype='uint8').reshape((1, 1, 3))
                cv2_color_space = {'rgb': None, 'hsv': cv2.COLOR_RGB2HSV, 'cie-lab': cv2.COLOR_RGB2LAB}
                if cv2_color_space.get(self.colorspace, None) is not None:
                    color = cv2.cvtColor(rgb, cv2_color_space[self.colorspace])
                    self.colors[col] = color
                else:
                    self.colors[col] = rgb

        if len(unk_colors) > 0:
            unmet_requirements.append(_("There are unsupported collors in the color list: {unk_colors}".format(
                unk_colors=str(unk_colors))))

        # bug: if i set the colors to be selected to "red" red is always in the result list,
        # even if I set min_color_threshold to 99.0

        # Make sure that we have all appropriate screenshots
        missing_screenshots = set()
        time_len = 0
        for anno in self.source_type.annotations:
            # create timestamps per annotation
            for i, timestamp in enumerate(range(anno.fragment.begin, anno.fragment.end)):
                if i == 0:
                    if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                        missing_screenshots.add(timestamp)
                elif timestamp % self.image_timestamp_divider == 0 or timestamp == int(
                        (anno.fragment.begin + anno.fragment.end) / 2):
                    if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                        missing_screenshots.add(timestamp)
                    time_len += 1
                elif i == len(range(anno.fragment.begin, anno.fragment.end)):
                    if self.controller.get_snapshot(annotation=anno, position=timestamp).is_default:
                        missing_screenshots.add(timestamp)
        if len(missing_screenshots) > 0:
            unmet_requirements.append(_("%d / %d screenshots are missing. Wait for extraction to complete.") % (
                len(missing_screenshots), time_len * len(self.source_type.annotations)))
        return unmet_requirements

    def iterator(self):
        """Iterate over the created annotations.
        """
        self.progress(.1, "Sending request to server")
        self.source_type = self.controller.package.get_element_by_id(self.source_type_id)
        new_atype = self.ensure_new_type(
                "domcols_%s" % self.source_type_id,
                title=_("Dominant Colors for %s" % self.source_type_id))
        new_atype.mimetype = 'text/plain'
        # FIXME: replace with proper values
        new_atype.setMetaData(config.data.namespace, "representation", 'here/content/parsed/label')
        
        def get_scaled_image(t):
            """Return the image at the appropriate scale for the selected model.
            """
            original = bytes(self.controller.package.imagecache.get(t))
            im = PIL.Image.open(BytesIO(original))
            im.save('/tmp/{0}.png'.format(t), 'PNG')
            if self.image_scale:
                im = im.resize((self.image_scale, self.image_scale))
            pixbuf = np.asarray(im, dtype='uint8')[:, :, :3]

            cv2_color_space = {'rgb': None, 'hsv': cv2.COLOR_RGB2HSV, 'cie-lab': cv2.COLOR_RGB2LAB}
            if cv2_color_space.get(self.colorspace, None) is not None:
                pixbuf = cv2.cvtColor(pixbuf, cv2_color_space[self.colorspace])
            return pixbuf

        def map_color_name(query):
            dists = dict()
            for c in self.colors:
                dists[c] = scipy.spatial.distance.sqeuclidean(self.colors[c]/255., query/255.)

            # sort dists ascending - color with smallest dist to query color is the most likely hit
            sorted_dists = sorted(dists.items(), key=lambda kv: kv[1])
            return sorted_dists[0]

        def extract_dominant_colors(frame_list):
            logger.info('Running color extraction')

            NUM_CLUSTERS = 10
            whitening = True

            pixels = []
            for frame in frame_list:  # traverse the given fragment
                im = frame.reshape((frame.shape[0] * frame.shape[1], 3))  # flatten the image to 1d
                pixels.extend(list(im))

            # 1. Variant - using median cut algorithm
            # Send array to quantize function which clusters values
            # using median cut algorithm
            #cmap = MMCQ.quantize(pixels, NUM_CLUSTERS)
            #dom_cols = cmap.palette
            #logger.info("dom cols: {0}".format(dom_cols))

            # 2. Variant - using kmeans clustering and
            obs = np.asarray(pixels, dtype='float32')
            if whitening:
                std_dev = obs.std(axis=0)
                obs = obs/std_dev

            # finding clusters
            centroids, meandist = kmeans(obs, NUM_CLUSTERS)
            nns, dist = vq(obs, centroids)  # assign codes, nns: row index of nearest centroid, dist: distance of pixel to nearest centroid
            counts, _ = scipy.histogram(nns, NUM_CLUSTERS)  # count occurrences
            dom_col_idx = np.argsort(counts)[::-1]

            dom_col_size = [counts[i]for i in dom_col_idx]
            dom_col_dist = [np.mean(dist[nns==i]) for i in dom_col_idx]
            dom_cols = [centroids[i] for i in dom_col_idx]

            if whitening:
                # undo whitening
                dom_cols *= std_dev

            dom_cols = dom_cols.astype('uint8')

            cnames = list()
            for c in dom_cols:
                cname = map_color_name(np.asarray(c).reshape((1,1,3)))
                logger.info("{0}  ({1})".format(cname[0], cname[1]))
                if not cname[0] in cnames:
                    cnames.append("{0}".format(cname[0]))

            return cnames

        response = {
            "model": 'standard',
            'media_uri': 'self.package.uri',
            'media_filename': self.controller.get_default_media(),
            'annotations': []}

        # iterate over annotations and extract dom colors per annotation
        for anno in self.source_type.annotations:
            frame_list = []
            for i, timestamp in enumerate(range(anno.fragment.begin, anno.fragment.end)):
                if i == 0:
                    frame_list.append(get_scaled_image(timestamp))
                elif timestamp % self.image_timestamp_divider == 0 or timestamp == int(
                        (anno.fragment.begin + anno.fragment.end) / 2):
                    frame_list.append(get_scaled_image(timestamp))
                elif i == len(range(anno.fragment.begin, anno.fragment.end)):
                    frame_list.append(get_scaled_image(timestamp))

            annotations = {'annotationid': anno.id,
                           'begin': anno.fragment.begin,
                           'end': anno.fragment.end,
                           'dominant_colors': extract_dominant_colors(frame_list)}
            response['annotations'].append(annotations)

        # create and yield dom colors annotations
        output = json.dumps(response)

        progress = .2
        step = .8 / (len(output) or 1)
        self.progress(.2, _("Parsing %d results") % len(output))

        for anno in response['annotations']:
            # a = self.package.get_element_by_id(anno['annotationid'])
            yield {
                'type': new_atype,
                'begin': anno['begin'],
                'end': anno['end'],
                'content': ",".join(anno['dominant_colors'])}
            self.progress(progress)
            progress += step
