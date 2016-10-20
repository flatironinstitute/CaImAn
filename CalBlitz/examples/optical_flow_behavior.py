# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:31:55 2016
OPTICAL FLOW
@author: agiovann
"""
cd '/home/agiovann/Dropbox (Simons Foundation)/Eyeblink/Datasets/AG051514-01/060914 C/-34 555 183_COND_C_001/-34 555 183_COND_C_X25/tr_X25'
#%%
import ca_source_extraction as cse
import calblitz as cb
import numpy as np
import pylab as pl
#%%
from scipy.io import loadmat
import cv2
mmat=loadmat('mov_AG051514-01-060914 C.mat')['mov']
m=cb.movie(mmat.transpose((2,0,1)),fr=120)
#%%
m.play(backend='opencv',magnification=4)
#%% dense flow: select only region that you think contains important information. Important, there is a considerable border effect 
fig=pl.figure()
pl.imshow(m[0],cmap=pl.cm.gray)
pts = fig.ginput(0, timeout=0)
data = np.zeros(np.shape(m[0]), dtype=np.int32)
pts = np.asarray(pts, dtype=np.int32)
cv2.fillConvexPoly(data, pts, (1,1,1), lineType=cv2.LINE_AA)
#data=np.float32(data)
pl.close()
#%%
#numstdthr=4.
#vect_diff=np.array(np.mean(np.abs(np.diff(m*data,axis=0)),axis=(1,2)))
#thresh=np.mean(vect_diff)+numstdthr*np.std(vect_diff)
#pks=scipy.signal.find_peaks_cwt(vect_diff,widths=np.arange(1,4),gap_thresh=2,max_distances=[100, 100])
#idx_bad=np.nonzero(vect_diff>thresh)
#
#for idx in idx_bad[0]:
##    print idx
#    m[idx]=np.mean(m[idx-4:idx+4],axis=0)
frate=120
prvs=np.uint8(m[0])
frame1=cv2.cvtColor(prvs, cv2.COLOR_GRAY2RGB)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255


cv2.namedWindow( "frame2", cv2.WINDOW_NORMAL )
# flow pars
pyr_scale=.1
levels=3 
winsize=25
iterations=3
poly_n=7
poly_sigma=1.5
mpart=m#[16300:16800]
T,d1,d2=mpart.shape
angs=np.zeros(T)
mags=np.zeros(T)
mov_tot=[]
do_show=False
do_write=False
#data[np.nonzero(data==0)]=np.nan
mov_tot=np.zeros([T,2,d1,d2])

if do_write:
    video = cv2.VideoWriter('video_2.avi',cv2.VideoWriter_fourcc('M','J','P','G'),30,(d2*2,d1),1)

for counter,next_ in enumerate(mpart):
    print counter          
    frame2 = cv2.cvtColor(np.uint8(next_), cv2.COLOR_GRAY2RGB)
    
#    if np.mean(np.abs(next_-prvs))<thresh:
    flow = cv2.calcOpticalFlowFarneback(prvs,next_, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
#    else:
#        print 'setting flow to zeros since possibly large frame change (trial break?)'
#        flow=np.zeros([d1,d2,2])
    
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    mags[counter]=np.sum(mag*data*frate)/np.sum(data)
    angs[counter]=np.sum(ang/2./np.pi*360*data)/np.sum(data)
    mag*=data
    ang*=data
    
    
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    frame_tot=np.concatenate([rgb,frame2],axis=1)
    mov_tot[counter,0]=mag
    mov_tot[counter,1]=ang
#    mov_tot[counter,0]=flow[:,:,0]
#    mov_tot[counter,1]=flow[:,:,1]
    if do_write:
        video.write(frame_tot)
    if do_show:
        cv2.imshow('frame2',frame_tot)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    prvs = next_

if do_write:
    video.release()

cv2.destroyAllWindows()

pl.plot(mags*10),pl.plot(angs)
pl.legend(['speed (pixels/second)','angle (degrees)'])
#%% NMF
isnmf=0
import time
tt=time.time()
from sklearn.decomposition import NMF
#if isnmf:
#    nmf=NMF(n_components=6)
#    newm=np.reshape(mov_tot,(T,d1*d2*2))
#    time_trace=nmf.fit_transform(newm[1:,:4800])
#else:
nmf=NMF(n_components=6)
newm=np.concatenate([mov_tot[:,0,:,:],mov_tot[:,1,:,:]],axis=0)
newm=np.reshape(mov_tot,(2*T,d1*d2),order='F') 
time_trace=nmf.fit_transform(newm)

el_t=time.time()-tt
print el_t
spatial_filter=nmf.components_
#%%
pl.figure()
count=0
for comp,tr in zip(spatial_filter,time_trace.T):
    count+=1    
    pl.subplot(6,2,count)
    pl.imshow(np.reshape(comp,(d1,d2),order='F'))
    count+=1    
    
    pl.subplot(6,2,count)
    if isnmf:    
        pl.plot(tr)
    else:
        pl.plot(np.reshape(tr,(T,2),order='F')/np.array([1, 1])+np.array([0,.3]))

#%%


#%% single featureflow,. seems not towork 
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
   
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                  

color = np.random.randint(0,255,(100,3))
old_gray=m[0]
old_frame = cv2.cvtColor(old_gray, cv2.COLOR_GRAY2RGB)

p0 = cv2.HoughCircles(old_gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
p0=p0.transpose(1,0,2)[:,:,:-1]
p1 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
cv2.namedWindow( "Display window", cv2.WINDOW_NORMAL );
for counter,frame_gray in enumerate(m[16000:17000]):
    print counter
    frame=cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow("Display window",img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
