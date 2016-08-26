%%
clear all
clc
% fls=dir('*morphology.mat');
% 'k31_20160104_MMA_150um_65mW_zoom2p2', ...

fl_templ={'k31_20151223_AM_150um_65mW_zoom2p2_00001_1-14',...
'k31_20160106_MMA_400um_118mW_zoom2p2_00001_1-19', ...
'k31_20160107_MMP_150um_65mW_zoom2p2_00001_1-15', ...
'k36_20151229_MMA_200um_65mW_zoom2p2_00001_1-17', ...
'k36_20160115_RSA_400um_118mW_zoom2p2_00001_20-38', ...
'k36_20160127_RL_150um_65mW_zoom2p2_00002_22-41', ...
'k37_20160109_AM_150um_65mW_zoom2p2_00001_1-16'};

fl_morph={'k31_20151223_AM_150um_65mW_zoom2p2_00001.cnmf-proto-roi-posthoc.morphology.mat',...
    'k31_20160106_MMA_400um_118mW_zoom2p2_00001.cnmf-proto-roi-posthoc.morphology.mat',...
    'k31_20160107_MMP_150um_65mW_zoom2p2_00001.cnmf-proto-roi-posthoc.morphology.mat',...
    'k36_20151229_MMA_200um_65mW_zoom2p2_00001.cnmf-proto-roi-posthoc.morphology.mat',...
    'k36_20160115_RSA_400um_118mW_zoom2p2_00001.cnmf-proto-roi-posthoc.morphology.mat',...
    'k36_20160127_RL_150um_65mW_zoom2p2_00002.cnmf-proto-roi-posthoc.morphology.mat',...
    'k37_20160109_AM_150um_65mW_zoom2p2_00001.cnmf-proto-roi-posthoc.morphology.mat'}

fl_morph_id={1,1,1,1,2,2,1};

for files_idx=1:numel(fl_templ)
    disp(files_idx)
    
    fls=subdir(['*' fl_templ{files_idx} '*.mat']);
    
    fname_masks=[];
    for kkk=1:numel(fls)
        if strfind(fls(kkk).name,'proto-roi.mat') & isempty(strfind(fls(kkk).name,'biased'))
            disp(fls(kkk).name)
            fname_masks=fls(kkk).name;
            break
        end
    end
    base_folder=fileparts(fname_masks);
    load(fname_masks)
%     load(fullfile(base_folder, roiFile{mf}),'cnmf')
    
    A=cnmf.spatial;
    C=cnmf.temporal;    
    ImageSize=cnmf.region.ImageSize;
    yRange=source.cropping.yRange;
    xRange=source.cropping.xRange;
    A_in=source.prototypes;
    template=source.fileMCorr.reference;
    selectMask=source.cropping.selectMask;
    load(fullfile(base_folder,fl_morph{files_idx}))
    roiFile=morphology(fl_morph_id{files_idx}).roiFile;
    disp(roiFile)
    globalId=morphology(fl_morph_id{files_idx}).globalID;
    init=int32(morphology(fl_morph_id{files_idx}).init);
    name_shapes=morphology(fl_morph_id{files_idx}).init.all();
    
    save([fname_masks(1:end-4) '_python.mat' ],'roiFile','globalId','init','name_shapes', ...
                                'A','C','A_in','template','yRange','xRange','ImageSize','selectMask')
                            
    clear roiFile globalId init name_shapes  A C A_in template selectMask ImageSize yRange xRange

end    
%%
fls=subdir('*python.mat');
for k=1:numel(fls)
    copyfile(fls(k).name,'.')
end
% %%
% fls=subdir('*morphology.mat')
% for kk = 1:numel(fls)
%     disp(fls(kk).name)
%     base_folder=fileparts(fls(kk).name);
%     load(fls(kk).name)
%     load([fls(kk).name(1:end-38) '.proto-roi.mat'])
%     roiFile={};
%     globalId={};
%     init={};
%     A_s={};
%     C_s={};
%     A_ins={};
%     templates={};
%     name_shapes={};
%     for mf = 1: numel(morphology)
%         try
%             roiFile{mf}=morphology(mf).roiFile;        
%             disp('loading')
%             load(fullfile(base_folder, roiFile{mf}),'cnmf')
%             A_s{mf}=cnmf.spatial;
%             C_s{mf}=cnmf.temporal;
% 
%             globalId{mf}=morphology(mf).globalID;
% 
%             init{mf}=int32(morphology(mf).init);
%             name_shapes{mf}=morphology(mf).init.all();
% 
%             A_ins{mf}=full(prototypes(mf).spatial);
%             templates{mf}=prototypes(mf).registration.reference;
%         catch
%             disp('****** FAILED *******')
%             disp(fullfile(base_folder, roiFile{mf}))
%         end
%     end
%     disp('saving')
%     save([fls(kk).name(1:end-4) '_python.mat' ],'roiFile','globalId','init','name_shapes','A_s','C_s','A_ins','templates')
%     clear roiFile globalId init name_shapes A_s C_s A C A_ins template
% end
% %
% % A=cnmf.spatial;
% % C=cnmf.temporal;
% % S=cnmf.spiking;
% % 
