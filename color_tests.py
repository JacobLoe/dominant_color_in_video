import cv2
import json
import numpy as np
from scipy.spatial import cKDTree


def load_color_palette(fname):
   return json.load(open(fname,'r'))


def load_rgbimg(fname):
   img = cv2.imread(fname)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   return img


def quantize_img(img, colors):
   values = np.asarray(colors, dtype=np.uint8)
   index = cKDTree(values)
   color_indexes = np.zeros(img.shape[:2], dtype=np.uint8)
   for r in range(img.shape[0]):
      for c in range(img.shape[1]):
         color_indexes[r][c] = index.query(img[r][c])[1]

   return color_indexes


if __name__ == '__main__':
   rgb_palette = list(load_color_palette('colors_full.json').values())
   rgb_img = load_rgbimg('pictures/original.png')
   color_indexes = quantize_img(img=rgb_img, colors=rgb_palette)
   qimg = np.zeros(rgb_img.shape, dtype=np.uint8)
   for r in range(qimg.shape[0]):
      for c in range(qimg.shape[1]):
         qimg[r][c] = np.uint8(rgb_palette[color_indexes[r][c]])
   cv2.imwrite('nearest_rgb.png', cv2.cvtColor(qimg, cv2.COLOR_RGB2BGR))

   lab_palette = [list(cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_RGB2LAB)[0][0]) for v in rgb_palette]
   ab_palette = [v[1:] for v in lab_palette]
   lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
   ab_img = lab_img[:,:,1:]
   color_indexes = quantize_img(img=ab_img, colors=ab_palette)
   qimg = np.zeros(lab_img.shape, dtype=np.uint8)
   for r in range(qimg.shape[0]):
      for c in range(qimg.shape[1]):
         qimg[r][c] = np.uint8(lab_palette[color_indexes[r][c]])
   cv2.imwrite('nearest_ab.png', cv2.cvtColor(qimg, cv2.COLOR_LAB2BGR))

   hsv_palette = [list(cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_RGB2HSV)[0][0]) for v in rgb_palette]
   hs_palette = [v[:2] for v in hsv_palette]
   hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
   hs_img = hsv_img[:,:,:2]
   color_indexes = quantize_img(img=hs_img, colors=hs_palette)
   qimg = np.zeros(hsv_img.shape, dtype=np.uint8)
   for r in range(qimg.shape[0]):
      for c in range(qimg.shape[1]):
         qimg[r][c] = np.uint8(hsv_palette[color_indexes[r][c]])
   cv2.imwrite('nearest_hs.png', cv2.cvtColor(qimg, cv2.COLOR_HSV2BGR))

   ycrcb_palette = [list(cv2.cvtColor(np.uint8([[v]]), cv2.COLOR_RGB2YCrCb)[0][0]) for v in rgb_palette]
   ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YCrCb)
   color_indexes = quantize_img(img=ycrcb_img, colors=ycrcb_palette)
   qimg = np.zeros(ycrcb_img.shape, dtype=np.uint8)
   for r in range(qimg.shape[0]):
      for c in range(qimg.shape[1]):
         qimg[r][c] = np.uint8(ycrcb_palette[color_indexes[r][c]])
   cv2.imwrite('nearest_ycrcb.png', cv2.cvtColor(qimg, cv2.COLOR_YCrCb2BGR))
