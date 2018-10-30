%% script to transform the output of Suite2p into output usable for CaImAn. 
%% you need to run the master_file_example.m script before
addpath('/opt/local/Data/Example/') % add the path to your make_db file
make_db_example; % RUN YOUR OWN MAKE_DB SCRIPT TO RUN HERE
db0 = db;
counter = 0;
nSVDforROI = 500:500:1500;
NavgFramesSVD = 2000:2000:6000;
sig = 0.25:0.25:0.75;
for nSVD = nSVDforROI
    disp('*')
    for Navg = NavgFramesSVD
        for ss = sig
            disp(nSVD)
            disp(Navg)
            disp(ss)
            counter = counter + 1;

            base_folder = ['/opt/local/Data/Example/DATA/F_' num2str(nSVD) '_' num2str(Navg) '_' num2str(ss) '/'];
            for iexp = 1:length(db0)

                db = db0(iexp);    
                fname = fullfile(base_folder,db.mouse_name,db.date,'1','*_plane1.mat');
                disp(fname)
                filelist = dir(fname);
                load(fullfile(base_folder,db.mouse_name,db.date,'1',filelist.name));

                dims = size(ops.mimg');
                traces = Fcell{1};
                nr = size(traces,1);
                T = size(traces,2);
                iscell = zeros(nr,1);
                masks = zeros(dims(1),dims(2),nr);
                disp(size(masks))
                for ii = 1:nr
                   st = stat(ii);
                   mask = zeros(dims);
                   mask(sub2ind(dims,st.xpix,st.ypix)) = st.lam;
                   masks(:,:,ii) = mask;   
                   iscell(ii)=st.iscell;
                end
                masks = single(masks);
                save(fullfile(base_folder,db.mouse_name,db.date,'1','python_out.mat'),'dims','traces','masks','iscell','-v7.3')

            end
        end
    end
end

%%
