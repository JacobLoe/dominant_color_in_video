#####################################################
## libraries
#####################################################
import extract_dominant_color
import numpy as np
######################################
#def test_bins_to_df():
#    
#    pd.DataFrame()
#    assert bins_to_df()==
def test_frames():
    black=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
                [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])
    red=np.array([[[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
             [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]],
              [[255,0,0],[255,0,0],[255,0,0],[255,0,0],[255,0,0]]])
    frames_black=[black,black,black]
    frames_red=[black,red,black]
#    assert frames_black[0].all()==frames_red[0].all()
    assert frames_black[1].all()==frames_red[1].all()
if __name__ == "__main__":
   print('unit_tests')
