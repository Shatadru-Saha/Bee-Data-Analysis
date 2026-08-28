import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
# import cv2
import pickle
import glob
# import psycopg2
import gzip
import scipy.stats


## 2019 info
# Note that 2019 x and y coordinates needs to be divided by 1.16 in order to match the comb map. This is done in 'df_to_coords', which should be called before getting substrate location

# This needs to be called and set just after import, and before using the package.  It sets which version of 'definitions' to use
def init(bd_input):  
    global bd
    bd = bd_input

################### Misc useful functions
# converts indexing by tagid, to indexing by uid.  Note that then later have to filter by age
def sel_cohort_bee(beedataarray):
    # input:  length of 4096
    # output:  selected output for cohort bee ids, selecting all (b/c later will select for age)
    # note that this is only valid for data on a certain day, because otherwise 
    return np.concatenate([beedataarray[ids] for ids in bd.cohort_tagids])

def convert_tagids_to_uids(tagidlist,daynum):
    # input:  list of tagids, e.g. [0,200, ...]
    # output:  list of uids, e.g. [492, 3002, ...].
    # if the bee should not be born yet on this day, returns -1.  Note that this does NOT check for death - that needs to be done separately.
    tag_to_uid = sel_cohort_bee(np.arange(bd.numbees))
    ages = np.concatenate(getages(daynum))
    found = [np.where( (tag_to_uid==tid) & (ages>=0))[0] for tid in tagidlist]
    return np.array([(f[0] if len(f)>0 else -1) for f in found])

# takes timestampts, and converts them to integers
def assign_integer_framenums(times):
    sectimes = np.array([pd.Timestamp(t).hour*3600 + pd.Timestamp(t).minute*60 + pd.Timestamp(t).second + pd.Timestamp(t).microsecond/10**6 for t in times])
    return np.floor(sectimes*3).astype(int)

def assign_integer_framenums_hourminsec(hour,minute,second):
    # second can be a float
    return np.floor( (hour*3600 + minute*60 + second)*3 ).astype(int)

def flat_to_hist(flatrow):
    # assume that the hist is at the end of the row
    numhistbins = bd.numxbins*bd.numybins
    return np.reshape(np.array(flatrow)[-numhistbins:],(bd.numxbins,bd.numybins))

##########################################################################################################################################
### SUBSTRATE AND COMB RELATED FUNCTIONS
# These are used to create a 'comb' class, that reads in image(s) (pixels) - a single day if measurement occurred on that day, or two days if the day is between measurements, converts to substrate map which has integer values based on  assignments, and functions to use this (getsubstrate, getsubstrate_simple
##########################################################################################################################################    

# USE THE ALREADY PICKLED FILES INSTEAD OF THIS
# def get_comb_images(comb_contents_dir,meas_num):
#     if bd.year==2018:
#         cam0 = glob.glob(comb_contents_dir+'start'+str(meas_num).zfill(2)+'_680_674*')[0]  # entrance side
#         cam1 = glob.glob(comb_contents_dir+'start'+str(meas_num).zfill(2)+'_220_219*')[0]  # other side
#     if bd.year==2019:
#         cam0 = np.flip(np.sort(glob.glob(comb_contents_dir+'entrance*png')))[meas_num]
#         cam1 = np.flip(np.sort(glob.glob(comb_contents_dir+'nonentrance*png')))[meas_num]
#     cam0img = cv2.imread(cam0)
#     cam1img = cv2.imread(cam1)
#     return cam0img, cam1img


def get_closest_measurements(daynum):
    # input:  day
    # output:  list containing either one or two entries (measurement numbers) with the combs for that day, 
    #          and the number of days between the current day and comb measurement (daydiffs)
    dd = daynum
    prevmeas = np.where(dd-bd.comb_daynums>=0)[0]
    m1 = [prevmeas[-1]] if len(prevmeas)>0 else []
    nextmeas = np.where(dd-bd.comb_daynums<=0)[0]
    m2 = [nextmeas[0]] if len(nextmeas)>0 else []
    measnums = np.unique(np.concatenate((m1,m2))).astype(int)
    daydiffs = np.abs(dd-bd.comb_daynums[measnums])
    return measnums, daydiffs

def image_to_substrate_map(img):
    # input:  image data
    # output:  array the same size as img, but containing integer for substrate identification
    # make a blank map of the same size as the image, and then fill in numbers for the different BGR values

    color_map = np.zeros(img.shape[0:2]).astype(int)
    
    # set values in the colormap
    threshold_sum = 20
    for i in range(len(bd.color_list)):
        distance = np.sum(np.abs(img - bd.color_list[i][np.newaxis,np.newaxis,:]),axis=-1)
        color_map[distance<=threshold_sum] = i+1
        # previous way:  searched for equality
#         color_map[np.all(img==bd.color_list[i],axis=2)] = i+1  # set to i+1, because in the next step, check for zeros

    # while loop to get rid of all the 0 values:
    num_unassigned = np.sum(color_map==0)
    while num_unassigned > 0:
        #print(num_unassigned)
        un_assigned_xs = np.where(color_map == 0)[0]
        un_assigned_ys = np.where(color_map == 0)[1]

        # randomly add up to a certain amount of pixels, to reassign
        shift=5
        shift_xs = un_assigned_xs + np.random.randint(-shift,shift+1,num_unassigned)     
        shift_ys = un_assigned_ys + np.random.randint(-shift,shift+1,num_unassigned) 
        shift_xs = np.minimum(np.maximum(shift_xs,0),color_map.shape[0]-1)
        shift_ys = np.minimum(np.maximum(shift_ys,0),color_map.shape[1]-1)    

        # change the unassigned x's and y's to the shifted values (5 px away)
        color_map[un_assigned_xs, un_assigned_ys] = color_map[shift_xs, shift_ys]        
        num_unassigned = np.sum(color_map==0)    
    
    # now, since there are no zeros left, subtract 1 when return it
    return color_map-1



# Class for storing comb data and manipulating it
class day_comb_data:
    def __init__(self,comb_contents_dir,day_input):
        # comb_contents_dir is a string
        # daynum 
        if type(day_input)==type(bd.alldaytimestamps[0]):
            daynum = np.where(day_input==bd.alldaytimestamps)[0][0]
        else:
            daynum = day_input
        
        # import the comb images
        measnums, daydiffs = get_closest_measurements(daynum)
        if len(measnums)==1:
            weights = np.array([1])
        else:
            # treat the special case where there were two measurements in a day
            if np.all(daydiffs==0):  # two measurements in a single day - then use the first one
                weights = np.array([1,0])
            else:
                weights = 1/daydiffs
                weights = weights/np.sum(weights)
        comb_images = [get_comb_images(comb_contents_dir,c) for c in measnums]
        # convert these to "substrate mappings"
        substrate_maps_raw = [[image_to_substrate_map(c) for c in cday] for cday in comb_images]
        # the images aren't all the same size.  Resize by padding with zeros for a given day
        sizes = [[c.shape for c in cday] for cday in substrate_maps_raw]
        maxsizes = [np.maximum(s[0],s[1]) for s in sizes]
        default_label = 8 if bd.year==2018 else 14 # label as 'white' for the points outside of the image
        substrate_maps = [np.ones(np.concatenate([[2],m]))*default_label for m in maxsizes]
        for d in range(len(measnums)):
            for c in range(2):
                sm = substrate_maps_raw[d][c]
                substrate_maps[d][c,0:sm.shape[0],0:sm.shape[1]] = sm   
        
        # save variables, for use
        self.substrate_maps = substrate_maps
        self.nummaps = len(measnums)
        self.measnums = measnums
        self.daydiffs = daydiffs
        self.weights = weights

    def getsubstrate(self,cameranums,c1,c2):  # coordinate numbers are switched
        maxx = np.min([c.shape[2] for c in self.substrate_maps])-1
        maxy = np.min([c.shape[1] for c in self.substrate_maps])-1
        c2_int = np.round(c2).astype(int)
        c1_int = np.round(c1).astype(int)
        return [m[cameranums,np.minimum(c2_int,maxy),np.minimum(c1_int,maxx)] for m in self.substrate_maps]


    
def getsubstrate_simple(comb,dancecomb,cameranums,c1,c2):
    subs = comb.getsubstrate(cameranums,c1,c2)
    dancesubs = dancecomb.getsubstrate(cameranums,c1,c2)
    for i in range(comb.nummaps):
        # honey is the same
        # brood:  2,1 -> 1
        subs[i][(subs[i]==1)|(subs[i]==2)] = 1
        # Empty comb: 3 -> 2
        subs[i][subs[i]==3] = 2
        # Pollen:  4 -> 3
        subs[i][subs[i]==4] = 3
        # Dance floor (dancecomb 5) -> 4
        subs[i][dancesubs[0]==5] = 4
        # other -> 5
        subs[i][subs[i]>4] = 5
    return subs
    

    
##########################################################################################################################################
#### DATABASE and RAW DATA RELATED FUNCTIONS
# Function to query the database and return a pandas df, then to sort, delete duplicates, and return np arrays
##########################################################################################################################################    


# This function takes the return from the above function, sorts its, uses a higher confidence if desired, deletes any duplicates, and returns np arrays 
def df_to_coords(df,conf_threshold=0.8):  # using a higher confidence threshold here
    # sort by timestamp, and secondary by confidence
    framenums = assign_integer_framenums(df['timestamp'])
    df['framenum'] = framenums
    df.sort_values(['framenum', 'bee_id_confidence'], ascending=[True, True], inplace=True)
    # if there is a duplicate timestamp for a bee, keep the last one (which will have the higher confidence)
    df_nodup = df.drop_duplicates(subset=['bee_id', 'framenum'],keep='last')
    
    conf = np.array(df_nodup['bee_id_confidence'])
    sel = (conf>conf_threshold) 
    if len(sel)>0:
        times = np.array(df_nodup['timestamp'][sel],dtype="datetime64[ns]")
        camera = np.array(df_nodup['cam_id'])[sel]    
        x = np.array(df_nodup['x_pos'])[sel]
        y = np.array(df_nodup['y_pos'])[sel]
        orientation = np.array(df_nodup['orientation'])[sel]
        bee_ids = np.array(df_nodup['bee_id'])[sel]
        framenums = (np.array(df_nodup['framenum']).astype(np.uint64))[sel]  # removed, because 2019 data does not have
        conf = conf[sel]
        if bd.year==2019:  # do the scaling for 2019, to align with the comb maps.
            conv_factor = 1.16
            x = x/conv_factor
            y = y/conv_factor
        return camera, x, y, orientation, bee_ids, times, framenums, conf     
    else:
        return [], [], [], [], [], [], [], []

    
##########################################################################################################################################
## CALCULATIONS
# multiple calculations that are useful for data reduction
##########################################################################################################################################
    
def fixanglerange(angles):
    return np.arctan2(np.sin(angles),np.cos(angles))

# Flattens the coordinates by ignoring the 'camera' dimension,  and making a reflection of the xpixels so it folds up (like it is in real life)
def shift_and_flatten_pixels(x,y,camera):
    x_flat = x.copy()
    y_flat = y.copy()
    leftcam = (camera==1)
    x_flat[leftcam] = bd.xpixels - x_flat[leftcam]
    y_flat[leftcam] = y_flat[leftcam] + bd.leftimage_yshift
    return x_flat, y_flat

# used for calculating dispersion - distance
def getflatdistance(x,y,camera):
    x_flat, y_flat = shift_and_flatten_pixels(x,y,camera)
    return np.sqrt(np.diff(x_flat)**2 + np.diff(y_flat)**2)

# returns counts of which 'frame' of the observation hive a bees was on
def getframehist(x,y,camera):
    vals_r = np.histogram(y[camera==0],bins=[0,bd.div1_r,bd.div2_r,bd.ypixels])[0] 
    vals_l = np.histogram(y[camera==1],bins=[0,bd.div1_l,bd.div2_l,bd.ypixels])[0]
    return np.array([vals_l,vals_r]) 

# returns the values for the frame histogram, just considering the right side - this is used for frame crossings
def getframehistvalues(x,y,camera):
    return np.digitize(y,[0,bd.div1_r,bd.div2_r,2*bd.ypixels]) - 1 + 3*(camera==0) 

# return x-y histogram, using the bins and edges that are set in definitions
def getxyhist(x,y,camera):
    x_adjusted = x + (np.logical_not(camera).astype(int))*bd.xpixels  # camera 0 is on the right, camera 1 on the left
    hist = np.histogram2d(x_adjusted,y,bins=[bd.x_edges,bd.y_edges])[0]    
    return hist

# returns velocity vector 'histogram', conditional on the x-y position
def getvelhist(x,y,camera):
    x_adjusted = x + (np.logical_not(camera).astype(int))*bd.xpixels  # camera 0 is on the right, camera 1 on the left
    dcamera = np.diff(camera)    
    dx = np.diff(x_adjusted)
    dy = np.diff(y)
    dcsel = dcamera == 0
    if np.sum(dcsel)>0:    
        velhist = scipy.stats.binned_statistic_2d(x_adjusted[1:][dcsel],y[1:][dcsel],[dx[dcsel],dy[dcsel]],bins=[bd.x_edges,bd.y_edges])[0]     
    else:
        velhist = np.tile(np.nan,(bd.numxbins,bd.numybins))        
    return velhist

# returns speed 'histogram', which is speed conditional on x-y position
def getspeedhist(x,y,camera):
    x_adjusted = x + (np.logical_not(camera).astype(int))*bd.xpixels  # camera 0 is on the right, camera 1 on the left
    dcamera = np.diff(camera)
    dx = np.diff(x_adjusted)  # it doesnt' matter if use x or x_adjusted, because select so that remove points where camera is the same
    dy = np.diff(y)
    spd = np.sqrt(dx**2+dy**2)
    dcsel = dcamera == 0
    if np.sum(dcsel)>0:
        velhist = scipy.stats.binned_statistic_2d(x_adjusted[1:][dcsel],y[1:][dcsel],spd[dcsel],bins=[bd.x_edges,bd.y_edges])[0]      
    else:
        velhist = np.tile(np.nan,(bd.numxbins,bd.numybins))
    return velhist

# get exits distance as the distance that would need to be traveled to reach the exit, by ignoring the distance to cross from one side to the other, but correcting for the lower left frame
def getexitdistance(x,y,camera):  # this isn't very efficient, because its shifts twice, but its cleaner to reuse functions than to have the same thing twice
    x_exit, y_exit = bd.xpixels, bd.ypixels  # lower right of image
    x_flat, y_flat = shift_and_flatten_pixels(x,y,camera)
    #  shift the y pixels for ones in the lower left bin, because they can't go down to the exit
    sel = (y>bd.div2_l) & (camera==1)
    # note:  using the right div value, because its already the shifted coordinates
    y_flat[sel] = 2*bd.div2_r - y_flat[sel]

    return np.sqrt( (x_flat-x_exit)**2 + (y_flat-y_exit)**2 )

# get age of all bees, all cohorts, on this day
# returns list that contains per-cohort ages, or -1 if not born yet, i.e. [[a,a,a,a,..],[b,b,b,b], ... [-1,-1,...] ].  Flattening this list corresponds to indexing
def getages(daynum):
    ages = [(-1*np.ones(len(ids))).astype(int) for ids in bd.cohort_tagids]
    # calculate age for each
    for cnum in range(len(bd.cohort_tagids)): 
        c_age = (bd.startday - bd.cohort_birthdates[cnum]).days + daynum
        if c_age>=0:
            ages[cnum] = np.ones(len(ages[cnum])) * c_age
            # check if they haven't been reused yet.  other than the queen
            if cnum>0:
                reused = (daynum>=bd.cohort_lastuse_perbee[cnum])
                ages[cnum][reused] = -1
    return ages

####################################################################################################
# Functions
###### SUBSTRATE AND INSIDE/OUTSIDE FUNCTIONS ##################################
############################################################    

def get_inout_estimates(dfday, obs_threshold=5, exitdistthreshold=1000,numtimedivs=288):  # dfday = dataframe containing data for one day
    day_uids = np.unique(dfday['Bee unique ID']).astype(int) 
    bee_obs = np.tile(np.nan,(len(day_uids),numtimedivs))
    bee_exitdist = np.tile(np.nan,(len(day_uids),numtimedivs))
    dfids = np.array(dfday['Bee unique ID']).astype(int)
    day_ages = np.tile(-1,len(day_uids))

    for j,beeid in enumerate(day_uids):
        sel = (dfids==beeid)

        dfsel = dfday[sel].copy()
        day_ages[j] = dfsel['Age'].astype(int).values[0]
        td = dfsel['timedivision'].astype(int)
        bee_obs[j,td] = dfsel['Num. observations']
        bee_exitdist[j,td] = dfsel['Exit distance (median)']
        bee_exitdist[j,np.isnan(bee_obs[j])] = np.nan

    all_inhive = np.tile(np.nan,(len(day_uids),numtimedivs))

    for beenum in range(len(day_uids)):
        obs = bee_obs[beenum]>=obs_threshold
        closetoexit = bee_exitdist[beenum]<exitdistthreshold

        bins = np.append(np.insert(np.where(np.abs(np.diff(obs).astype(int)))[0]+1,0,0),numtimedivs)
        sections = np.array([bins[0:-1],bins[1:]]).T
        # each section has the same values, all True or all False.
        # set to 'in hive', where all are above the threshold
        if len(sections)==1: # special case of all are the same
            if obs[0]:
                all_inhive[beenum,:] = 1
            # if not, dont say anything, because don't know, this bee could be dead.
        else:
            for j,s in enumerate(sections):
                if obs[s[0]]:
                    all_inhive[beenum,s[0]:s[1]] = 1
                else:
                    if j==0: # treat the first section different
                        if closetoexit[np.min((s[1]+1,numtimedivs-1))]:
                            all_inhive[beenum,s[0]:s[1]] = 0
                        else:
                            all_inhive[beenum,s[0]:s[1]] = 1
                    else:
                        if closetoexit[s[0]-1]:    
                            all_inhive[beenum,s[0]:s[1]] = 0
                        else:

                            all_inhive[beenum,s[0]:s[1]] = 1

    return  day_uids, day_ages, all_inhive, bee_obs, bee_exitdist



def get_onsubstrate(dfday, obs_threshold=5, topfraction_threshold=0.5, substratename='Festoon',numtimedivs=288 ):  # dfday = dataframe containing data for one day
    day_uids = np.unique(dfday['Bee unique ID']).astype(int) 
    bee_obs = np.tile(np.nan,(len(day_uids),numtimedivs))
    bee_topframe = np.tile(np.nan,(len(day_uids),numtimedivs))
    bee_data = np.tile(np.nan,(len(day_uids),numtimedivs,dfday.shape[-1]))

    dfids = np.array(dfday['Bee unique ID']).astype(int)

    day_ages = np.tile(-1,len(day_uids))
    
    for j,beeid in enumerate(day_uids):
        sel = (dfids==beeid)

        dfsel = dfday[sel].copy()
        day_ages[j] = dfsel['Age'].astype(int).values[0]
        td = dfsel['timedivision'].astype(int)
        bee_obs[j,td] = dfsel['Num. observations']
        if substratename=='topframe':
            bee_topframe[j,td] = dfsel['Frame 0'] + dfsel['Frame 3']
        else:    
            bee_topframe[j,td] = dfsel[substratename]
        
        bee_data[j,td] = dfsel
        bee_topframe[j,np.isnan(bee_obs[j])] = np.nan
        bee_data[j,np.isnan(bee_obs[j])] = np.nan
        
    all_ontop = np.tile(np.nan,(len(day_uids),numtimedivs))

    for beenum in (range(len(day_uids))):
        obs = bee_obs[beenum]>=obs_threshold
        mostlyontop = bee_topframe[beenum]>topfraction_threshold

        bins = np.append(np.insert(np.where(np.abs(np.diff(obs).astype(int)))[0]+1,0,0),numtimedivs)
        sections = np.array([bins[0:-1],bins[1:]]).T
        # each section has the same values, all True or all False.
        # set to 'in hive', where all are above the threshold
        if len(sections)==1: # special case of all are the same
            if obs[0]&mostlyontop[0]:
                all_ontop[beenum,:] = True
            # if not, dont say anything, because don't know, this bee could be dead.
        else:
            for j,s in enumerate(sections):
                if obs[s[0]]:  # if observed in this section
                    all_ontop[beenum,s[0]:s[1]] = mostlyontop[s[0]]  # mark as ontop if above threshold
                else:  # if not observed
                    if j==0: # treat the first section different
                        all_ontop[beenum,s[0]:s[1]] = mostlyontop[s[1]]  # if the next segment has them on top, mark as on top
                    else:
                        all_ontop[beenum,s[0]:s[1]] = mostlyontop[s[0]-1] # if prev segment has on top, mark as on top

    return  day_uids, day_ages, all_ontop, bee_obs, bee_data



