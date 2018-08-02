## dominant_color_in_video

### performance

- with resolution (640,360) one frame takes 127 seconds to process
- with resolution (240,135) one frame takes 17 seconds to process
- with resolution (120,90) one frame takes 5 seconds to process
- with resolution (60,45) one frame takes 1.5 seconds to process

### dominant colors at resolution 120x90
#### tested on 4 different video-files, either every 8th or every 32th frame was taken

#### file1
##### every 8th frame
- color             count      
- purple4           126620
- purple            127525
- dim_grey          313349
- black             316265
- dark_olive_green  369125 

##### every 32th frame
- color             count     
- purple4           31730
- purple            31785
- dim_grey          78266
- black             84429
- dark_olive_green  92967

#### file2
##### every 8th frame
- color             count
- silver            193991
- light_grey        219070
- dark_khaki        227012
- dark_olive_green  260499
- black             289723 

##### every 32th frame
- color             count
- silver            48332
- light_grey        55091
- dark_khaki        57389
- dark_olive_green  66190
- black             76940

#### file3
##### every 8th frame
- color             count
- darkred         39529
- royal_blue      49887
- olive_drab      55881
- black          112909
- saddle_brown  1181221

##### every 32th frame
- color             count
- darkred         9568
- royal_blue     12633
- olive_drab     13968
- black          33369
- saddle_brown  296496

#### file4
##### every 8th frame
- color             count
- silver          94572
- dark_blue      107089
- light_grey     136388
- antique_white  189913
- sandy_brown    713678

##### every 32th frame
- color             count
- silver          24527
- dark_blue       26919
- light_grey      33782
- antique_white   47885
- sandy_brown    179470