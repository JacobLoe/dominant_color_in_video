#####################################################
## libraries
#####################################################
import extract_dominant_color as edc
import numpy as np
#from nose import with_setup
######################################
def test_frames_black():
    black=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])
    frames=[black,black,black]
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 1.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_color(frames)
def test_frames_mixed():
    red=np.array([[[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
             [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]])
    blue=np.array([[[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]])
    frames=[red,blue,red,blue]
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 1.0,'blue': 0.5,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.5,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    print(edc.extract_dominant_color(frames))
    assert target == edc.extract_dominant_color(frames)
def test_fn_rgb_to_color():
    assert 1==1
def test_bins_to_df():
    assert 1==1
def test_read_video_segments_output_dims():
    assert np.shape(edc.read_video_segments('videos/red.mp4',0,9,5))==(10,3,5,3)
if __name__ == "__main__":
   print('unit_tests')
