#####################################################
## libraries
#####################################################
import extract_dominant_color as edc
import numpy as np
#from nose import with_setup
###########################################################################################################################################################



############################################################################################
#def test_bins_to_df():
#    assert 1==0
############################################################################################
#def test_azp_path()
#    assert 1==0
############################################################################################
# read_video_segments
############################################################################################
def test_read_video_segments_output_dims():
    assert np.shape(edc.read_video_segments('videos/red.mp4',0,9,5,'RGB'))==(9,3,5,3)
############################################################################################
# extract_dominant_colors
############################################################################################
def test_frames_segment_rgb_black():
    black=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],dtype='uint8')
    frames=[black,black,black]
    frames=edc.change_colorspace(frames,'rgb')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 1.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'rgb','full','segment')

def test_frames_segment_rgb_white():
    white=np.array([[[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]],dtype='uint8')
    frames=[white,white,white]
    frames=edc.change_colorspace(frames,'rgb')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 1.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'rgb','full','segment')

def test_frames_segment_rgb_mixed():
    red=np.array([[[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]],dtype='uint8')
    blue=np.array([[[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]],dtype='uint8')
    frames=[red,blue,red,blue]
    frames=edc.change_colorspace(frames,'rgb')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.5,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.5,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'rgb','full','segment')

def test_frames_segment_hsv_black():
    black=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],dtype='uint8')
    frames=[black,black,black]
    frames=edc.change_colorspace(frames,'HSV')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 1.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'HSV','full','segment')

def test_frames_segment_hsv_white():
    white=np.array([[[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]],dtype='uint8')
    frames=[white,white,white]
    frames=edc.change_colorspace(frames,'HSV')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 1.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'HSV','full','segment')

def test_frames_segment_hsv_mixed():
    red=np.array([[[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]],dtype='uint8')
    blue=np.array([[[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]],dtype='uint8')
    frames=[red,blue,red,blue]
    frames=edc.change_colorspace(frames,'HSV')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.5,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.5,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'HSV','full','segment')


def test_frames_segment_cielab_black():
    black=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],dtype='uint8')
    frames=[black,black,black]
    frames=edc.change_colorspace(frames,'cie-lab')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 1.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'cie-lab','full','segment')

def test_frames_segment_cielab_white():
    white=np.array([[[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]],dtype='uint8')
    frames=[white,white,white]
    frames=edc.change_colorspace(frames,'cie-lab')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 1.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'cie-lab','full','segment')

def test_frames_segment_cielab_mixed():
    red=np.array([[[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]],dtype='uint8')
    blue=np.array([[[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]],dtype='uint8')
    frames=[red,blue,red,blue]
    frames=edc.change_colorspace(frames,'cie-lab')
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.5,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.5,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(frames,'cie-lab','full','segment')


def test_frames_scene_rgb_black():
    black=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]],dtype='uint8')
    frames=[black,black,black]
    frames=edc.change_colorspace(frames,'rgb')
    scene =[frames,frames]
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 1.0,'blue': 0.0,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.0,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(scene,'rgb','full','scene')

def test_frames_scene_rgb_mixed():
    red=np.array([[[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]],dtype='uint8')
    blue=np.array([[[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]],dtype='uint8')
    frames=[red,blue,red,blue]
    frames=edc.change_colorspace(frames,'rgb')
    scene =[frames,frames]
    target ={'antiquewhite': 0.0,'aquamarine': 0.0,'black': 0.0,'blue': 0.5,'coral': 0.0, 'crimson': 0.0, 'cyan': 0.0, 'darkblue': 0.0,'darkcyan': 0.0, 'darkgreen': 0.0, 'darkkhaki': 0.0, 'darkolivegreen': 0.0,'darkorange': 0.0, 'darkred': 0.0, 'deeppink': 0.0, 'dimgrey': 0.0,'firebrick': 0.0, 'gold': 0.0, 'goldenrod': 0.0, 'green': 0.0,'greenyellow': 0.0, 'grey': 0.0, 'ivory': 0.0, 'khaki': 0.0,
 'lightgrey': 0.0, 'magenta': 0.0, 'olivedrab': 0.0, 'orange': 0.0,'pink': 0.0, 'purple': 0.0, 'purple4': 0.0, 'red': 0.5,'royalblue': 0.0, 'saddlebrown': 0.0, 'salmon': 0.0, 'sandybrown': 0.0,'silver': 0.0, 'skin': 0.0, 'skyblue': 0.0, 'steelblue': 0.0,'tomato': 0.0, 'violet': 0.0, 'wheat': 0.0, 'white': 0.0, 'yellow': 0.0}
    assert target == edc.extract_dominant_colors(scene,'rgb','full','scene')
############################################################################################
# change_colorspace
############################################################################################
def test_change_colorspace_rgb():
    colors=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[128,128,128],[128,128,128],[128,128,128],[128,128,128],[128,128,128]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]],dtype='uint8')
    target=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[128,128,128],[128,128,128],[128,128,128],[128,128,128],[128,128,128]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]],dtype='uint8')
    target_list=[target,target]
    frames=[colors,colors]
    frames_changed=edc.change_colorspace(frames,'rgb')
    assert np.array_equal(frames_changed,target_list)

def test_change_colorspace_hsv():
    colors=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[128,128,128],[128,128,128],[128,128,128],[128,128,128],[128,128,128]],
              [[255,255,255],[255,255,255],[255,255,255],[255,255,255],[255,255,255]]],dtype='uint8')
    target=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
              [[0,255,255],[0,255,255],[0,255,255],[0,255,255],[0,255,255]],
              [[120,255,255],[120,255,255],[120,255,255],[120,255,255],[120,255,255]],
              [[0,0,128],[0,0,128],[0,0,128],[0,0,128],[0,0,128]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]]],dtype='uint8')
    target_list=[target,target]
    frames=[colors,colors]
    frames_changed=edc.change_colorspace(frames,'HSV')
    assert np.array_equal(frames_changed,target_list)

def test_change_colorspace_lab():
    colors=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[0,0,255],[0,0,255],[0,0,255],[0,0,255],[0,0,255]],
              [[128,128,128],[128,128,128],[128,128,128],[128,128,128],[128,128,128]]],dtype='uint8')
    target=np.array([[[0,128,128],[0,128,128],[0,128,128],[0,128,128],[0,128,128]],
              [[136,208,195],[136,208,195],[136,208,195],[136,208,195],[136,208,195]],
              [[82,207,20],[82,207,20],[82,207,20],[82,207,20],[82,207,20]],
              [[137,128,128],[137,128,128],[137,128,128],[137,128,128],[137,128,128]]],dtype='uint8')
    target_list=[target,target]
    frames=[colors,colors]
    frames_changed=edc.change_colorspace(frames,'cie-lab')
    assert np.array_equal(frames_changed,target_list)
############################################################################################
# fn_rgb_to_color
#############################################################################################
def test_reduced_colors():
    full_colors_dict=edc.fn_rgb_to_color('rgb','colors')
    target_colors_dict={}
    with open('colors') as file:   
         for line in file:
             target_colors_dict[line]=0 
    assert len(full_colors_dict)==len(target_colors_dict)

def test_full_colors_rgb():
    full_colors_dict=edc.fn_rgb_to_color('rgb','full')
    target_colors_dict={(0, 0, 0): 'black',
(0, 0, 139): 'darkblue',
(0, 0, 255): 'blue',
(0, 100, 0): 'darkgreen',
(0, 128, 0): 'green',
(0, 139, 139): 'darkcyan',
(0, 255, 255): 'cyan',
(128, 128, 128): 'grey',
(65, 105, 225): 'royalblue',
(70, 130, 180): 'steelblue',
(85, 107, 47): 'darkolivegreen',
(105, 105, 105): 'dimgrey',
(107, 142, 35): 'olivedrab',
(127, 255, 212): 'aquamarine',
(128, 0, 128): 'purple',
(135, 206, 235): 'skyblue',
(139, 0, 0): 'darkred',
(139, 69, 19): 'saddlebrown',
(147, 112, 219): 'purple4',
(173, 255, 47): 'greenyellow',
(178, 34, 34): 'firebrick',
(189, 183, 107): 'darkkhaki',
(192, 192, 192): 'silver',
(211, 211, 211): 'lightgrey',
(218, 165, 32): 'goldenrod',
(220, 20, 60): 'crimson',
(238, 130, 238): 'violet',
(240, 230, 140): 'khaki',
(244, 164, 96): 'sandybrown',
(245, 222, 179): 'wheat',
(250, 128, 114): 'salmon',
(250, 235, 215): 'antiquewhite',
(255, 0, 0): 'red',
(255, 0, 255): 'magenta',
(255, 20, 147): 'deeppink',
(255, 99, 71): 'tomato',
(255, 127, 80): 'coral',
(255, 140, 0): 'darkorange',
(255, 165, 0): 'orange',
(255, 192, 203): 'pink',
(255, 215, 0): 'gold',
(255, 224, 189): 'skin',
(255, 255, 0): 'yellow',
(255, 255, 240): 'ivory',
(255, 255, 255): 'white'}
    assert full_colors_dict==target_colors_dict

def test_rgb_to_color_lab():
    full_colors_dict=edc.fn_rgb_to_color('cie-lab','full')
    target_colors_dict={(0, 128, 128): 'black',
 (38, 178, 59): 'darkblue',
 (72, 179, 170): 'darkred',
 (76, 187, 91): 'purple',
 (82, 207, 20): 'blue',
 (92, 85, 170): 'darkgreen',
 (96, 154, 169): 'saddlebrown',
 (100, 184, 165): 'firebrick',
 (108, 109, 159): 'darkolivegreen',
 (113, 128, 128): 'dimgrey',
 (118, 76, 178): 'green',
 (120, 199, 162): 'crimson',
 (122, 154, 63): 'royalblue',
 (124, 101, 120): 'grey',
 (133, 97, 119): 'darkcyan',
 (134, 124, 96): 'steelblue',
 (136, 208, 195): 'red',
 (139, 100, 178): 'olivedrab',
 (140, 165, 78): 'purple4',
 (143, 212, 122): 'deeppink',
 (154, 226, 67): 'magenta',
 (159, 186, 174): 'tomato',
 (172, 173, 157): 'salmon',
 (172, 173, 175): 'coral',
 (177, 165, 203): 'darkorange',
 (178, 184, 91): 'violet',
 (181, 137, 197): 'goldenrod',
 (187, 119, 167): 'darkkhaki',
 (189, 151, 175): 'sandybrown',
 (191, 152, 207): 'orange',
 (198, 128, 128): 'silver',
 (202, 113, 107): 'skyblue',
 (213, 152, 131): 'pink',
 (216, 128, 128): 'lightgrey',
 (222, 126, 215): 'gold',
 (228, 130, 152): 'wheat',
 (231, 119, 173): 'khaki',
 (232, 134, 149): 'skin',
 (233, 80, 114): 'cyan',
 (235, 76, 210): 'greenyellow',
 (235, 82, 138): 'aquamarine',
 (239, 130, 140): 'antiquewhite',
 (248, 106, 223): 'yellow',
 (254, 125, 135): 'ivory',
 (255, 128, 128): 'white'}
    assert full_colors_dict==target_colors_dict

def test_rgb_to_color_hsv():
    full_colors_dict=edc.fn_rgb_to_color('HSV','full')
    target_colors_dict={(0, 0, 0): 'black',
 (0, 0, 105): 'dimgrey',
 (0, 0, 192): 'silver',
 (0, 0, 211): 'lightgrey',
 (0, 0, 255): 'white',
 (0, 206, 178): 'firebrick',
 (0, 255, 139): 'darkred',
 (0, 255, 255): 'red',
 (3, 139, 250): 'salmon',
 (5, 184, 255): 'tomato',
 (8, 175, 255): 'coral',
 (13, 220, 139): 'saddlebrown',
 (14, 155, 244): 'sandybrown',
 (16, 66, 255): 'skin',
 (16, 255, 255): 'darkorange',
 (17, 36, 250): 'antiquewhite',
 (19, 255, 255): 'orange',
 (20, 69, 245): 'wheat',
 (21, 218, 218): 'goldenrod',
 (25, 255, 255): 'gold',
 (27, 106, 240): 'khaki',
 (28, 111, 189): 'darkkhaki',
 (30, 15, 255): 'ivory',
 (30, 255, 255): 'yellow',
 (40, 192, 142): 'olivedrab',
 (41, 143, 107): 'darkolivegreen',
 (42, 208, 255): 'greenyellow',
 (60, 255, 100): 'darkgreen',
 (60, 255, 128): 'green',
 (80, 128, 255): 'aquamarine',
 (90, 199, 128): 'grey',
 (90, 255, 139): 'darkcyan',
 (90, 255, 255): 'cyan',
 (99, 109, 235): 'skyblue',
 (104, 156, 180): 'steelblue',
 (113, 181, 225): 'royalblue',
 (120, 255, 139): 'darkblue',
 (120, 255, 255): 'blue',
 (130, 125, 219): 'purple4',
 (150, 116, 238): 'violet',
 (150, 255, 128): 'purple',
 (150, 255, 255): 'magenta',
 (164, 235, 255): 'deeppink',
 (174, 232, 220): 'crimson',
 (175, 63, 255): 'pink'}
    assert full_colors_dict==target_colors_dict
############################################################################################
if __name__ == "__main__":
   print('this does unit_tests')
