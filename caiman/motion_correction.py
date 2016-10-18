## -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 21:02:12 2016

@author: agiovann, adapted from motion correction algorithm from Selmaan Chettih
"""
#%%
import calblitz as cb
import numpy as np
import pylab as pl
#%%
import scipy
chone=cb.load('/Users/agiovann/Documents/MATLAB/Motion_Correction/M_FLUO_1.tif',fr=15)
chone=chone[:,8:-8,8:-8]
T=np.median(chone,axis=0)
Nbasis = 8
minIters = 5

#linear b-splines
knots = np.linspace(1,np.shape(T)[0],Nbasis+1);
knots = np.hstack([knots[0]-(knots[1]-knots[0]),knots,knots[-1]+(knots[-1]-knots[-2])]);

weights=knots[:-2]
order=len(knots)-len(weights)-1

x=range(T.shape[0])

B = np.zeros((len(x),len(weights)))
for ii in range(len(knots)-order-1):
    B[:,ii] = bin(this.knots,ii,this.order,x);
end

spl = fastBSpline(knots,knots(1:end-2))


B = spl.getBasis((1:size(T,1))');
Tnorm = T(:)-mean(T(:));
Tnorm = Tnorm/sqrt(sum(Tnorm.^2));
B = full(B);

lambda = .0001*median(T(:))^2;
theI = (eye(Nbasis+1)*lambda);

Bi = B(:,1:end-1).*B(:,2:end);
allBs = [B.^2,Bi];
[xi,yi] = meshgrid(1:size(T,2),1:size(T,1)); 

#%%
def doLucasKanade_singleFrame(T, I, B, allBs, xi, yi, theI, Tnorm, nBasis=4, minIters=5):

    maxIters = 50
    deltacorr = 0.0005
    
    _ , w = np.shape(T)
    
    #Find optimal image warp via Lucas Kanade    
    c0 = mycorr(I(:), Tnorm);
    
    for ii = 1:maxIters
#        %Displaced template
#        Dx = repmat((B*dpx), 1, w);
#        Dy = repmat((B*dpy), 1, w);
        
        Id = interp2(I, xi, yi, 'linear', 0);
                
        %gradient
        [dTx, dTy] = imgradientxy(Id, 'centraldifference');
        dTx(:, [1, ]) = 0;
        dTy([1, ], :) = 0;
        
        if ii > minIters
            c = mycorr(Id(:), Tnorm);
            if c - c0 < deltacorr && ii > 1
                break;
            
            c0 = c;
        
 
        del = T - Id;
 
        %special trick for g (easy)
        gx = B'*sum(del.*dTx, 2);
        gy = B'*sum(del.*dTy, 2);
 
        %special trick for H - harder
        Hx = constructH(allBs'*sum(dTx.^2,2), nBasis+1) + theI;
        Hy = constructH(allBs'*sum(dTy.^2,2), nBasis+1) + theI;
 
        dpx = Hx\gx;
        dpy = Hy\gy;
        
    return [Id, dpx, dpy]

#         dpx = dpx + damping*dpx_;
#         dpy = dpy + damping*dpy_;
    


function thec = mycorr(A,B)
    meanA = mean(A(:));
    A = A(:) - meanA;
    A = A / sqrt(sum(A.^2));
    thec = A'*B;

 
function H2 = constructH(Hd,ns)
%     H2d1 = Hd(1:ns)';
%     H2d2 = [Hd(ns+1:);0]';
%     H2d3 = [0;Hd(ns+1:)]';
%     
%     if isa(Hd, 'gpuArray')
%         H2 = gpuArray.zeros(ns);
%     else
%         H2 = zeros(ns);
%     
%             
%     H2((0:ns-1)*ns+(1:ns)) = H2d1;
%     H2(((1:ns-1)*ns+(1:ns-1))) = H2d2(1:-1);
%     H2(((0:ns-2)*ns+(1:ns-1))+1) = H2d3(2:);

    if isa(Hd, 'gpuArray')
        H2 = gpuArray.zeros(ns);
    else
        H2 = zeros(ns);
    
            
    H2((0:ns-1)*ns+(1:ns)) = Hd(1:ns)';
    H2(((1:ns-1)*ns+(1:ns-1))) = Hd(ns+1:)';
    H2(((0:ns-2)*ns+(1:ns-1))+1) = Hd(ns+1:)';


def motion_correct_parallel(file_names,fr,template=None,margins_out=0,max_shift_w=5, max_shift_h=5,remove_blanks=False,apply_smooth=False,dview=None,save_hdf5=True):
    """motion correct many movies usingthe ipyparallel cluster
    Parameters
    ----------
    file_names: list of strings
        names of he files to be motion corrected
    fr: double
        fr parameters for calcblitz movie 
    margins_out: int
        number of pixels to remove from the borders    
    
    Return
    ------
    base file names of the motion corrected files
    """
    args_in=[];
    for file_idx,f in enumerate(file_names):
        if type(template) is list:
            args_in.append((f,fr,margins_out,template[file_idx],max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5))
        else:
            args_in.append((f,fr,margins_out,template,max_shift_w, max_shift_h,remove_blanks,apply_smooth,save_hdf5))
        
    try:
        
        if dview is not None:
#            if backend is 'SLURM':
#                if 'IPPPDIR' in os.environ and 'IPPPROFILE' in os.environ:
#                    pdir, profile = os.environ['IPPPDIR'], os.environ['IPPPROFILE']
#                else:
#                    raise Exception('envirnomment variables not found, please source slurmAlloc.rc')
#        
#                c = Client(ipython_dir=pdir, profile=profile)
#                print 'Using '+ str(len(c)) + ' processes'
#            else:
#                c = Client()


            file_res = dview.map_sync(process_movie_parallel, args_in)                         
            dview.results.clear()       

            

        else:
            
            file_res = map(process_movie_parallel, args_in)        
                 
        
        
    except :   
        
        try:
            if dview is not None:
                
                dview.results.clear()       

        except UnboundLocalError as uberr:

            print 'could not close client'

        raise
                                    
    return file_res 

#%%
####
### Load and Motion Correct
#
## This first part just creates a cell array of filenames to load, may have
## to be changed if you're names are different than Alice's
#
##get all files names
##%%
#num_files = 2
#nameOffset = 0
#mouse_name = 'AG01'
#session_name = '010101'
#view_name = 'view1'
#slice_name = 'slice1'
#
#
#
#filenames=['M_FLUO_1.tif','M_FLUO_2.tif']
#fr=15.
#n_blocks_x=3
#n_blocks_y=3
#
#
#max_shift_rigid=10
#max_shift_aff=3
#
#fullfilename = filenames[0]
#chone=cb.load(fullfilename,fr=fr)
##chone=chone.crop(5,5,5,5)
#chone,_,_,_=chone.motion_correct(max_shift_rigid,max_shift_rigid,remove_blanks=True)
##info=imfinfo(fullfilename)
#Z,N,M=chone.shape
#
#
##scale movie for seamless intensities
##Clip bad image region
#meanlastframes=np.median(np.mean(chone[-400:],axis=(1,2)))
#meanfirstframes=np.median(np.mean(chone[:400],axis=(1,2)))
#chone=chone*(meanlastframes/meanfirstframes)    
#
##Construct Movie Segments
#
#xind = np.floor(np.linspace(0,M/2.,n_blocks_x))
#yind = np.floor(np.linspace(0,N/2.,n_blocks_y))
#segPos = []
#for x in range(len(xind)):
#    for y in range(len(yind)):
#        segPos.append([xind[x], yind[y],  np.floor(M/2.), np.floor(N/2.)])
#    
#
#nSeg = len(segPos)
#segPos=np.asarray(segPos,dtype=np.int)
##First order motion correction
##clear xshifts yshifts corThresh,
#xshifts=[]
#yshifts=[]
#new_chone=chone.copy().resize(1,1,.5).resize(1,1,2)
#
#for Seg in range(nSeg):
#    print(Seg)
#    tMov = new_chone[:,segPos[Seg,1]:segPos[Seg,1]+segPos[Seg,3],segPos[Seg,0]:segPos[Seg,0]+segPos[Seg,2]].copy()
#    print(tMov.shape)
#    print(segPos[Seg,:])
##        tBase = np.percentile(tMov,1)
##        tTop = np.percentile(tMov,99)
##        tMov = (tMov - tBase) / (tTop-tBase)
##        tMov[tMov<0] = 0 
##        tMov[tMov>1] = 1
#    tMov,xsh,xcorr,templ=tMov.motion_correct(max_shift_w=max_shift_aff, max_shift_h=max_shift_aff)
#    xsh=np.array(xsh)
#    xshifts.append(xsh[:,0])
#    yshifts.append(xsh[:,1])
##        [xshifts(Seg,:),yshifts(Seg,:)]=track_subpixel_wholeframe_motion_varythresh(...
##        tMov,median(tMov,3),5,0.9,100)
#
#xshifts=np.array(xshifts)
#yshifts=np.array(yshifts)
#    
#       
#pl.plot(np.array(xshifts).T,color='r')
#pl.plot(np.array(yshifts).T,color='g')
##%%
#from skimage import data
#from skimage import transform as tf
#
#usePos = range(nSeg)
#
#yoff = segPos[usePos,1] + np.floor(segPos[usePos,3]/2.)
#xoff = segPos[usePos,0] + np.floor(segPos[usePos,2]/2.)
#rpts = np.concatenate([xoff[:,np.newaxis], yoff[:,np.newaxis]],axis=1)
#
#chone=(chone-np.min(chone))/(np.max(chone)-np.min(chone))
#new_mov=[]
#for frame,img in enumerate(chone):
#    if frame250 ==0:
#        print(frame)
#    
#    xframe = xshifts[usePos,frame] + xoff
#    yframe = yshifts[usePos,frame] + yoff
#    fpts=np.concatenate([xframe[:,np.newaxis], yframe[:,np.newaxis]],axis=1)
#    tform = tf.ProjectiveTransform()
#    tform.estimate(rpts, fpts)
#    warped = tf.warp(img, tform, output_shape=img.shape)
#    new_mov.append(warped)
##    tform=fitgeotrans(fpts,rpts,'affine')
##    cated_movie(:,:,frame)=imwarp(cated_movie(:,:,frame),tform,'OutputView',R)   
#new_mov=cb.concatenate([chone,cb.movie(np.array(new_mov),fr=15)],axis=1)
##%%
#
##cated_movie(:,:,1) = cated_movie(:,:,2)
#
#blank=sum(cated_movie==0,3)
#xmin=find(median(blank,1)==0,1,'first'),
#xmax=find(median(blank,1)==0,1,'last'),
#ymin=find(median(blank,2)==0,1,'first'),
#ymax=find(median(blank,2)==0,1,'last'),
#cated_movie=cated_movie(ymin:ymax,xmin:xmax,:)
#save(sprintf('#s_#s_#s_#s',mouse_name,session_name,view_name,slice_name),'cated_movie','chone_mask','-v7.3')
#
### Calculate piecewise covariance, principle components, and feed to ICA algorithm
#
#M=size(cated_movie,1)
#N=size(cated_movie,2)
#Z=size(cated_movie,3)
#nPCs = 1e3
#
#cated_movie=reshape(cated_movie,M*N,size(cated_movie,3))
#pxmean = mean(cated_movie,2)
#cated_movie = cated_movie ./ repmat(pxmean/100,1,Z)
#
#display('------------Computing Principle Components-------------')
#[e,s,l]=pca(cated_movie,'NumComponents',nPCs)
#
#covtrace = double(sum(l))
#CovEvals = double(l(1:nPCs))
#mixedsig= double(e(:,1:nPCs))'
#mixedfilters = double(reshape(s(:,1:nPCs),M,N,nPCs))
#save(sprintf('#s_#s_#s_#s_PCs',mouse_name,session_name,view_name,slice_name),'mixedsig','mixedfilters','covtrace','CovEvals')
#clear e s l
#
#PCuse = 1:300
#mu=.2
#nIC = ceil(length(PCuse)/1)
#ica_A_guess = []
#termtol = 1e-7
#maxrounds = 1e3
#smwidth = 3
#thresh = 2
#arealims = [20 400]
#plotting = 1
#[ica_sig, ica_filters, ica_A, numiter] = CellsortICA(...
#     mixedsig,mixedfilters, CovEvals, PCuse, mu, nIC, ica_A_guess, termtol, maxrounds)
##CellsortChoosePCs(shiftdim(ica_filters,1),M,N)
#[ica_segments, segmentlabel, segcentroid] = CellsortSegmentation...
#    (ica_filters, smwidth, thresh, arealims, plotting)
#for i=1:size(ica_segments,1)
#    normSeg(:,:,i)=100*ica_segments(i,:,:)/norm(reshape(ica_segments(i,:,:),1,[]))
#
#normSeg = reshape(normSeg,M*N,[])
#SegTraces = normSeg' * cated_movie
#normSeg = reshape(normSeg,M,N,[])
#
#for i=1:size(ica_segments,1)
#    segSize(i) = squeeze(sum(sum(ica_segments(i,:,:)>0)))
#    segSkew(i) = skewness(reshape(ica_segments(i,:,:),1,[]))
#
#segSTD = sqrt(std(SegTraces,[],2)./mean(SegTraces,2))
#segSize = zscore(segSize)
#segSkew = zscore(segSkew)
#
#goodSeg = find(segSize-segSkew > 0)
#gTrace = SegTraces(goodSeg,:)
#save(sprintf('#s_#s_#s_#s_ICs',mouse_name,session_name,view_name,slice_name),...
#    'SegTraces','normSeg','ica_sig','ica_filters','ica_A','ica_segments','segmentlabel','segcentroid',...
#    'segSize','segSkew','segSTD','goodSeg','gTrace')
#figure,scatter(segSize,segSkew,50,segSTD,'filled')