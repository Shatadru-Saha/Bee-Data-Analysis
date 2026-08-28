import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import definitions_2019 as bd
# import cv2

snscolors=sns.color_palette()
snscolors = np.concatenate((snscolors,snscolors))


# This needs to be called and set just after import, and before using the package.  It sets which version of 'definitions' to use
def init(bd_input):  
    global bd
    bd = bd_input 


#  creates a rgba image from histogram data, using a specified color and normalization
def rgba_cmap(histdata,normvalue=-1,color=[0,0,0]):
    new = np.zeros((histdata.shape[0],histdata.shape[1],4))
    if normvalue<0:
        normvalue = np.max(histdata)
    vals = 1 - np.minimum(histdata/normvalue,1)
    
    new[:,:,0] = color[0]
    new[:,:,1] = color[1]
    new[:,:,2] = color[2]
    new[:,:,3] = 1-vals
    return new



def createnewimage():
    f, ax = plt.subplots(1,1)
    f.set_size_inches(12,12)
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)            
    ax.set_xlim([0,2*bd.xpixels])
    ax.set_ylim([bd.ypixels,0])  # note this is backwards.. this makes it show 'zero' at the top, which is what shows for the comb, so its consistent (and no changes to data)    
    return f, ax

def showhist(hist,ax=[],color=[0,0,0],alpha=1,normvalue=-1):
    if ax==[]:
        f, ax = createnewimage()
    if normvalue==-1:
        normvalue=np.quantile(hist,0.99)

    rgba_img = rgba_cmap(hist.T,normvalue=normvalue,color=color)
    ax.imshow(rgba_img,extent=(0,2*bd.xpixels,bd.ypixels,0),alpha=alpha,rasterized=True)
#     ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
def showcomb(comb,ax=[],alphamult=1):
    if ax==[]:
        f, ax = createnewimage()
    for i in range(comb.nummaps):
        ax.imshow(np.hstack([comb.substrate_maps[i][1],comb.substrate_maps[i][0]]),cmap=bd.cmap_comb,norm=bd.cmap_norm,alpha=comb.weights[i]*alphamult,rasterized=True)
#     ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])    
    return ax
    
def showmonthday(timestamp):
    return str(timestamp.month).zfill(2)+'-'+str(timestamp.day).zfill(2)

    
        
def showframe(ax=[],color=snscolors[1]):  # the 'frame' was calculated and saved in 'Cohort 2d histograms'
    if ax==[]:
        f, ax = createnewimage()        
    ax.axvline(bd.xpixels,c=color,zorder=10)
    [ax.plot([0,bd.xpixels],[d,d],c=color,zorder=10) for d in [bd.div1_l,bd.div2_l]]
    [ax.plot([bd.xpixels,2*bd.xpixels],[d,d],c=color,zorder=10) for d in [bd.div1_r,bd.div2_r]]
    return ax
    
def plotbee_xy(x,y,camera,ax=[],color='k',s=10,alpha=0.7,joined=True,maxxydiff=80):
    if ax==[]:
        f, ax = createnewimage()     
    if len(x)>0:
        # leave this out - conversion is now in datafunctions, for df_to_coords.  
#         if bd.year==2019:
#             conv_factor = 1.16
        
        # convert to float, in case its a dataframe passed in
        x = np.array(x).astype(float)
        y = np.array(y).astype(float)
        camera = np.array(camera).astype(int)
        conv_factor = 1
        x_adjusted = x/conv_factor + (np.logical_not(camera).astype(int))*bd.xpixels
        if joined:
            # take parts when they were on the same camera, and join the trajectories
            xydiff = np.sqrt(np.diff(x_adjusted)**2 + np.diff(y/conv_factor)**2)
            splitcond = np.where( (xydiff > maxxydiff) | (np.abs(np.diff(camera))>0) )[0] + 1
            xtp = np.split(x_adjusted, splitcond)
            ytp = np.split(y/conv_factor, splitcond)
            for j in range(len(xtp)):
                if len(xtp[j])>1:
                    ax.plot(xtp[j],ytp[j],color=color,alpha=alpha,linewidth=s/10) 
        else:
            # plot as points (easier)
            ax.scatter(x_adjusted,y/conv_factor,color=color,s=s,alpha=alpha)
    return ax


# # MLS added this (from Jake Graving):

# def fig2data(fig):
#     """
#     @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
#     @param fig a matplotlib figure
#     @return a numpy 3D array of RGBA values
#     """
#     # draw the renderer
#     fig.canvas.draw()
 
#     # Get the RGBA buffer from the figure
#     w,h = fig.canvas.get_width_height()
#     buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
#     buf.shape = (h, w, 4)
 
#     buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    
#     return buf


# Cameras: 
# 674 - entrance side, bottom (dancefloor)
# 680 - entrance side, top
# 219 - non-entrance side, bottom
# 220 - non-entrance side, top 



###### DISPLAY - PCA related ##################################
############################################################

def pcacomponentplots(ax,vh,ylabels,colors=''):
    numev = vh.shape[0]
    xlim = 1.05*np.max(np.abs(vh))
    if colors=='':
        colors = np.tile([0.35,0.35,0.35],(numev,1))
    elif len(colors)<numev:
        colors = np.tile(colors,(numev,1))
    for evnum in range(numev):
        a=ax[evnum]
        x=vh[evnum]
        y = np.flipud(np.arange(vh.shape[1]))
    #     ax.plot(x,y,'-o',label='$\\vec e_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',c=snscolors[evnum])
        thickness=0.75
        a.barh(y+0*(evnum-1)*thickness,x,height=thickness,color=colors[evnum])
        a.set_yticks(y)
        a.set_yticklabels(ylabels,rotation='horizontal',fontsize=12)
        a.axvline(0,c='k',linestyle='--')
#         a.set_title(label='$\\vec v_'+str(evnum)+'$: '+str(np.round(pcavar[evnum]*100,1))+'%',fontsize=14)
        a.set_xlim([-xlim,xlim])

def plot_tsne_withcolors(ax,tsne_result,quantity,title,corrskip=1,plotskip=1,colortype='scalar',qmin=0.001,qmax=0.999,alphaval=0.3,s=4,coloroffset=0,cmapname='cool',setxylimquantile=False):
    colordata = quantity.copy()
    if len(colordata)>len(tsne_result):
        colordata = colordata[::corrskip]
    if len(colordata.shape)>1:
        colordata = colordata[:,0]
    if colortype=='scalar':
        cmap=plt.get_cmap(cmapname)  # or 'cool'
        q0,q1 = np.quantile(colordata,[qmin,qmax])
        colordata = colordata-q0
        colordata = colordata/(q1-q0)
        colordata[colordata<0] = 0
        colordata[colordata>1] = 1
        colordata *= 0.99
        colors = cmap(colordata)
    else:
#         cmap=plt.get_cmap('Set1')
        colors = snscolors[colordata.astype(int)+coloroffset]
    tp = tsne_result
    # [ax.scatter([-100],[-100],alpha=1,s=10,color=cmap(i*0.99/np.max(groupvalues)),label='group '+str(i)) for i in np.arange(max(groupvalues)+1)]  # legend hack
    scatterplot = ax.scatter(tp[::plotskip,0],tp[::plotskip,1],s=s,alpha=alphaval,color=colors[::plotskip],rasterized=True)
    if setxylimquantile:
        ax.set_xlim(np.quantile(tp[:,0],[qmin,qmax]))
        ax.set_ylim(np.quantile(tp[:,1],[qmin,qmax]))
    else:
        ax.set_xlim(np.quantile(tp[:,0],[0,1]))
        ax.set_ylim(np.quantile(tp[:,1],[0,1]))
    ax.set_title(title,fontsize=16)       
    return scatterplot, colordata    
    
def categorydists(n_clusters,membership,quantityvals,labels,pointskip=1,coloroffset=0,ax='',f='',npoints_in_title=False):
    numq = len(quantityvals)
    if len(ax)==0:
        f,ax = plt.subplots(1,n_clusters,sharex=True,sharey=False)
        f.set_size_inches(5*n_clusters,4)

    for i in range(n_clusters):
        a = ax[i]
        clr = snscolors[i+coloroffset] if coloroffset>=0 else 'k'
        sel = membership == i
        a.set_title('Cluster '+str(i+1)+(': '+str(np.sum(sel).astype(int))+' data points' if npoints_in_title else ''),fontsize=14)
        for j, q in enumerate(quantityvals):
            tp = q[sel]
            alpha_scaled = 0.2
            xval = len(quantityvals)-j-1
            bplot = a.boxplot(x=tp,positions=[xval],patch_artist=True,showfliers=False,showcaps=True,vert=False,widths=0.9)
            for patch in bplot['boxes']:
                patch.set(color=clr,alpha=alpha_scaled)  
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot[element], color=clr,alpha=1)     
            plt.setp(bplot["fliers"], markeredgecolor=clr, markerfacecolor=clr,markersize=4,alpha=alpha_scaled)
            xnoise=0.1
            xtp = np.random.normal(xval,xnoise,size=len(tp))
            a.scatter(tp[::pointskip],xtp[::pointskip],color=clr,alpha=0.03,zorder=10,s=3,rasterized=True)
    for a in ax:
        a.set_yticks(range(len(quantityvals)))
        a.set_yticklabels(np.flip(labels),rotation='horizontal',fontsize=12)
        a.axvline(0,c='k',linewidth=1)
        a.set_xlabel('Quantity value (std. dev from mean)',fontsize=12)
    return f,ax

def quantitydists(n_clusters,membership,quantityvals,labels,f='',ax='',pointskip=1,coloroffset=0,xorder=[],colorsel=[],color='k'):
    numq = len(quantityvals)
    if len(ax)==0:
        f,ax = plt.subplots(1,numq,sharex=True,sharey=True)
        f.set_size_inches(2*numq,3)

    if len(xorder)==0:
        xorder = np.arange(n_clusters)
    if len(colorsel)==0:
        colorsel=np.arange(n_clusters)
    # dmat_bygroup = [[ for n in range(n_clusters)] for qnum in range(dmat.shape[1])]
    for i in range(numq):
        a = ax[i]
        a.set_title(labels[i],fontsize=14)
        for j in range(n_clusters):
            if coloroffset>=100:
                clr = (color[j] if len(color)==n_clusters else color)
            else:
                clr = snscolors[colorsel[j]+coloroffset]
            tp = quantityvals[i][membership==j]
            alpha_scaled = 0.2
            xval = xorder[j]
            bplot = a.boxplot(x=tp,positions=[xval],patch_artist=True,showfliers=False,showcaps=True,vert=True,widths=0.9)
            for patch in bplot['boxes']:
                patch.set(color=clr,alpha=alpha_scaled)  
            for element in ['whiskers', 'fliers', 'medians', 'caps']:
                plt.setp(bplot[element], color=clr,alpha=1)     
            plt.setp(bplot["fliers"], markeredgecolor=clr, markerfacecolor=clr,markersize=4,alpha=alpha_scaled)
            xnoise=0.1
            xtp = np.random.normal(xval,xnoise,size=len(tp))
            a.scatter(xtp[::pointskip],tp[::pointskip],color=clr,alpha=0.03,zorder=10,s=3,rasterized=True)
    [a.set_xticks(range(n_clusters)) for a in ax]
    [a.set_xticklabels(np.arange(n_clusters)+1,fontsize=14) for a in ax]
    [a.tick_params(labelsize=14) for a in ax]
    ax[0].set_ylabel('Quantity value',fontsize=14)
    [a.set_xlabel('Cluster number',fontsize=14) for a in ax]
    [a.axhline(0,c='k',linewidth=1) for a in ax]
    return f, ax