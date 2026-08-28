# THIS IS FOR 2019 

import pandas as pd
import numpy as np
from matplotlib import colors

querydata = "dbname='beesbook' user='reader' host='localhost' password='reader' port='5435'"
databasename = 'bb_detections_2019_konstanz_complete'

year = 2019
startday = pd.Timestamp(year,6,5)  # actual, for 2019:   5 June 2019  (cameras turned on)
endday = pd.Timestamp(year,9,28)    # actual, for 2019:   28 September... but should be... 04 Oct (need to include Morgane recordings)
alldaytimestamps = pd.date_range(start=startday,end=endday,freq='D')
numdays = len(alldaytimestamps)  # total number of days is: 
numbees = 4096  #  I think 4096 is the total number.  This is the total number of barcodes, NOT the total number of bees actually tracked
numsubstrates = 16 # 16 actual substrates, and then 0 for undefined (updated for drone comb substrates in 2019)

# in 2019, measured the comb much more frequently: 
comb_daynums = np.array([0,1,2,3,4,5,6,7,8,12,13,14,15,16,17,18,19,20,21,22,23,27,27.5,33,34,34.5,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,68,72,75,78,82,86,89,92,95,99,103,106,114]) + 5 - startday.day  # days from start day, 5 June.  REMOVED days [121,148], because they are after the recording started
comb_daynums = comb_daynums.astype(int)
# 0.5 days, eg 27.5 and 34.5, are for when I traced comb contents, and then changed the combs and traced the new comb

# get bee cohort data, and just process these for a single one
cohort_data = pd.read_csv('all_cohorts_2019.csv')
cohort_colornames = np.unique(cohort_data['cohort'])
cohort_tagids = [np.array(cohort_data[cohort_data['cohort']==name]['beeID']) for name in cohort_colornames]
cohort_birthdates = np.array([pd.Timestamp(np.array(cohort_data['DOB'][cohort_data['cohort']==c])[0],freq='D') for c in cohort_colornames])
cohort_colonynames = np.array([cohort_data[cohort_data['cohort']==name]['colony'].iloc[0] for name in cohort_colornames])

numbeestracked = np.sum([len(l) for l in cohort_tagids])

cohortorder = np.argsort(cohort_birthdates)
# change sort order to be by birthdate
cohort_colornames = cohort_colornames[cohortorder]
cohort_tagids = [cohort_tagids[c] for c in cohortorder]
cohort_uids = [np.arange(len(cohort_tagids[0]))]
for c in range(1,len(cohort_tagids)):
    nextuid = cohort_uids[-1][-1] +1
    cohort_uids.append(np.arange(nextuid,nextuid+len(cohort_tagids[c])))
cohort_birthdates = cohort_birthdates[cohortorder]
# Hey Jacob - possible to match birthdates  to "numdays"? 
# This is how I'm doing it: ((bd.cohort_birthdates[8])-(bd.startday)).days.  Yea, thats how I'm doing it too.
cohort_colonynames = cohort_colonynames[cohortorder]

# define cohort names by phonetic alphabet (in 2019 its included in the table)
cohort_names = ['0_queen', '1_workers', '2_workers', '3_workers', '4_workers',
       'alpha', 'bravo', 'charlie', 'charlie_drones', 'delta',
       'delta_drones', 'echo', 'foxtrot', 'golf', 'hotel', 'indigo',
       'juliett', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papi',
       'quebec', 'romeo', 'sierra_0', 'sierra_1', 'sierra_2', 'tango',
       'unicorn', 'victory', 'whiskey', 'xray', 'yankee', 'zulu']

# loop through and get the 'last data date' for each bee.
cohort_lastuse_perbee = [((numdays+1)*np.ones(len(ids))).astype(int) for ids in cohort_tagids]
for cnum in range(len(cohort_tagids)): 
    # NOTE!  cohorts need to be sorted by birthday for this to work
    next_use = [cnum + 1+np.where([b in c for c in cohort_tagids[cnum+1:]])[0]
                for b in cohort_tagids[cnum]]
    used_again = np.array([len(l)>0 for l in next_use])
    for i in range(len(cohort_tagids[cnum])):
        if len(next_use[i])>0:
            cohort_lastuse_perbee[cnum][i] = ((cohort_birthdates[next_use[i][0]]-startday).days+1)
        
numcohorts = len(cohort_names)

# %ta
cohort_colors = [[0,0,0],  # queen is cohort 0
                 [0.12, 0.47, 0.71, 1.],
                 [0.68, 0.78, 0.91, 1.],
                 [1.00, 0.50, 0.05, 1.],
                 [1.00, 0.73, 0.47, 1.],
                 [0.17, 0.63, 0.17, 1.],
                 [0.60, 0.87, 0.54, 1.],
                 [0.84, 0.15, 0.16, 1.],
                 [0.58, 0.40, 0.74, 1.],
                 [0.77, 0.69, 0.83, 1.],
                 [0.55, 0.34, 0.29, 1.],
                 [0.77, 0.61, 0.58, 1.],
                 [0.89, 0.47, 0.76, 1.],
                 [0.50, 0.50, 0.50, 1.],
                 [0.78, 0.78, 0.78, 1.],
                 [0.74, 0.74, 0.13, 1.],
                 [0.09, 0.75, 0.81, 1.],
                [0,0,0],  # queen is cohort 0
                 [0.12, 0.47, 0.71, 1.],
                 [0.68, 0.78, 0.91, 1.],
                 [1.00, 0.50, 0.05, 1.],
                 [1.00, 0.73, 0.47, 1.],
                 [0.17, 0.63, 0.17, 1.],
                 [0.60, 0.87, 0.54, 1.],
                 [0.84, 0.15, 0.16, 1.],
                 [0.58, 0.40, 0.74, 1.],
                 [0.77, 0.69, 0.83, 1.],
                 [0.55, 0.34, 0.29, 1.],
                 [0.77, 0.61, 0.58, 1.],
                 [0.89, 0.47, 0.76, 1.],
                 [0.50, 0.50, 0.50, 1.],
                 [0.78, 0.78, 0.78, 1.],
                 [0.74, 0.74, 0.13, 1.],
                 [0.09, 0.75, 0.81, 1.],
                [0,0,0],  # queen is cohort 0
                 [0.12, 0.47, 0.71, 1.],
                 [0.68, 0.78, 0.91, 1.],
                 [1.00, 0.50, 0.05, 1.],
                 [1.00, 0.73, 0.47, 1.],
                 [0.17, 0.63, 0.17, 1.],
                 [0.60, 0.87, 0.54, 1.],
                 [0.84, 0.15, 0.16, 1.],
                 [0.58, 0.40, 0.74, 1.],
                 [0.77, 0.69, 0.83, 1.],
                 [0.55, 0.34, 0.29, 1.],
                 [0.77, 0.61, 0.58, 1.],
                 [0.89, 0.47, 0.76, 1.],
                 [0.50, 0.50, 0.50, 1.],
                 [0.78, 0.78, 0.78, 1.],
                 [0.74, 0.74, 0.13, 1.],
                 [0.09, 0.75, 0.81, 1.]]

# *** - "NA": all the things that didn't fit into any of the other colors... 

# 2019 colors: 
# these are ordered in B-G-R: 
# color_dict = {"yellow": [0, 255, 255], 
#              "darkblue":[146, 49, 46], 
#              "blue":[255, 0, 0], 
#              "green":[81, 166,   0], 
#              "orange":[34, 101, 242], 
#              "pink":[255,   0, 255], 
#              "brown":[36,  76, 117], 
#              "grey":[137, 137, 137], 
#              "black":[0, 0, 0], # this is for BLANK SPACE (building area)
#              "drone_yellow": [0, 215, 215], 
#              "drone_darkblue":[255, 93, 125], 
#              "drone_blue":[255, 149, 64], 
#              "drone_green":[121, 206,  40], 
#              "red":[36, 28, 237], # this is for FESTOON
#              "white":[255, 255, 255]               
#              }

# this is used for identifying colors in the images.  dont change!!
color_list = np.array([[0, 255, 255],  [146,  49,  46],  [255,   0,   0],  [81, 166,   0],  [34, 101, 242],  [255,   0, 255],  [36,  76, 117], [137, 137, 137], [0,0,0], [0, 215, 215], [255, 93, 125], [255, 149, 64], [121, 206, 40], [36, 28, 237], [255, 255, 255] ] )
# this is used for displaying and plotting:  can change # MLS added additional colors from color_list at the end here. 
color_list_display_full = np.array([[0, 255, 255],  [146,  49,  46],  [255,   0,   0],  [81, 166,   0],  [34, 101, 242],  [255,   0, 255],  [36,  76, 117], [137, 137, 137], [0,0,0], [0, 215, 215], [255, 93, 125], [255, 149, 64], [121, 206, 40], [36, 28, 237], [255, 255, 255] ] )
color_list_display = np.array([[0, 255, 255],  [146,  49,  46],  [255,   0,   0],  [81, 166,   0],  [34, 101, 242],  [255,   0, 255],  [36,  76, 117], [137, 137, 137], [0,0,0], [0, 215, 215], [255, 93, 125], [255, 149, 64], [121, 206, 40], [36, 28, 237], [255, 255, 255] ] )
color_list_rgb = np.fliplr(color_list_display)
color_names = ["yellow", "darkblue", 'blue', 'green', 'orange', 'pink', 'brown', 'grey', "black", "drone_yellow", "drone_darkblue", "drone_blue", "drone_green", "red", "white"]
color_labels = ['Honey','Capped brood','Young brood','Empty comb','Pollen stores','Dance floor','Wooden frames','Peripheral galleries', 'blank space', 'honey-dc', 'capped brood-dc', 'young brood-dc', 'empty comb-dc', 'festoon', 'white']
substrate_names = color_labels
substrate_names_simple = ['Honey','Brood','Empty comb','Pollen','Dance floor','Other']


# This creates a discrete colormap for showing the comb contents
cmap_comb = colors.ListedColormap(color_list_rgb/255)
cmap_bounds=np.arange(-0.5,len(color_list_rgb)+0.5)
cmap_norm = colors.BoundaryNorm(cmap_bounds, cmap_comb.N)

### Colors for plotting.  I made Michael's yellow a bit darker, # MLS added new colors for 2019: 
comb_color_palette = ["#CCCC00", "#2E3192", "#0000FF", "#00A651", "#F26522", "#FF00FF", "#754C24", "#898989", "#000000", "#D7D700", "#7D5DFF", "#4095FF", "#28CE79", "#ED1C26", "#FFFFFF",]

# name color, BGR code, contents, HEX codes: 
# "yellow": [0, 255, 255], honey, #FFFF00
# "darkblue":[146, 49, 46], capped brood, #2E3192
# "blue":[255, 0, 0], young brood, #0000FF
# "green":[81, 166,   0], empty comb, #00A651
# "orange":[34, 101, 242], pollen, #F26522
# "pink":[255,   0, 255], dancefloor, #FF00FF
# "brown":[36,  76, 117], wooden frames, #754C24
# "grey":[137, 137, 137], peripheral galleries, #898989
# "black":[0, 0, 0], blank space (comb building), #000000
# "drone_yellow": [0, 215, 215], honey in drone comb (dc), #D7D700
# "drone_darkblue":[255, 93, 125], capped brood in dc, #7D5DFF 
# "drone_blue":[255, 149, 64], young brood in dc, #4095FF
# "drone_green":[121, 206,  40], empty dc, #28CE79
# "red":[36, 28, 237], festoon (curtain of bees), #ED1C26
# "white":[255, 255, 255] white (undefined), #FFFFFF



#################################################################################################################################
######## Processing-related
#################################################################################################################################
# amount to shift down the left comb, so that it aligns better with the right one.
leftimage_yshift = 40  # from aligning the images, this looked reasonable.  did this to test:
    # test = comb.substrate_maps[0][1].copy()
    # test[leftimage_yshift:] = test[:-leftimage_yshift]
    # f,ax = plt.subplots(1,1)
    # f.set_size_inches(10,20)
    # plt.imshow(comb.substrate_maps[0][0],alpha=0.5)
    # plt.imshow(1-np.fliplr(test),alpha=0.5)
    # plt.suptitle(shift,fontsize=25)
    # plt.show()
    # flip the left one
    
# THIS IS FOR 2018: 
# ypixels, xpixels = (5652, 3296)  # comb.substrate_maps[0][0].shape    
# MLS edited this for 2019: 
ypixels, xpixels = (5720, 3296)  # comb.substrate_maps[0][0].shape    

## '6-frame representation' 
# the divs are for the one at the right.  for the 
div1_r = 1880+10
div2_r=1830*2+20
div1_l = div1_r - leftimage_yshift
div2_l = div2_r - leftimage_yshift

# Spatial histogram bins
pixels_per_bin = 160
pixels_per_cm = 80  # conversion factor from Michael
#  Michael says:  (variation between cameras wasn't that huge, from 78.3 to 80.7 px per cm)

numxbins = np.round(2*xpixels/pixels_per_bin).astype(int)
numybins = np.round(ypixels/pixels_per_bin).astype(int)
x_edges = np.linspace(0,2*xpixels,numxbins+1)
y_edges = np.linspace(0,ypixels,numybins+1)       

# manipulations done in 2019
expsummary = pd.read_csv('summary_experiments_2019.csv')
selexpdays = lambda x: np.array(expsummary[x]['exp_day'])
dronedays = selexpdays(expsummary['drones_introduced']==True)
heatdays = selexpdays(expsummary['heat_stress']==True)
# print(np.unique(expsummary['darwinian_demon']))
foragerblockdays_8am = selexpdays(expsummary['darwinian_demon']=='demon_8am')
foragerblockdays_12pm = selexpdays(expsummary['darwinian_demon']=='demon_12pm')
foragerblockdays_2hr = selexpdays(expsummary['darwinian_demon']=='2hr_demon')
foragerblockdays_reverse = selexpdays(expsummary['darwinian_demon']=='reverse_12pm_demon')
foragerblockdays_x2x2x = selexpdays(expsummary['darwinian_demon']=='x2x2x')
foragerblockdays_control = selexpdays(expsummary['darwinian_demon']=='control')
feederondays = selexpdays(expsummary['comb_feeder_on']==True)

combexchangedays = selexpdays(expsummary['exchange_comb']==True)