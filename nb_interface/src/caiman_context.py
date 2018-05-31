
import pickle
######

#@out.capture()()
def save_obj(path, obj):
    with open(path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#@out.capture()()
def load_obj(path):
    with open(path + '.pkl', 'rb') as f:
        return pickle.load(f)

#@out.capture()()
class Context: #used to save data related to analysis (not serializable)
	def __init__(self, cluster=[]):
		#setup cluster
		if cluster != []:
			self.c = cluster[0]
			self.dview = cluster[1]
			self.n_processes = cluster[2]
		else:
			self.c, self.dview, self.n_processes = None,None,None

		self.working_dir = ''
		self.working_mc_files = [] #str path
		self.working_cnmf_file = None  #str path
		self.mc_dsfactors = None  #downsample factors: [x,y,t]

		self.mc_mmaps = [] # list of file names of created mmaps from motion correction
		self.mc_rig = []    #rigid mc results
		self.mc_nonrig = [] #non-rigid mc results

		self.YrDT = None # tuple (Yr, dims, T), Yr: numpy array (memmory mapped file)
		self.cnmf_results = [] #A, C, b, f, YrA, sn, idx_components, S
		self.idx_components_keep = []  #result after filter_rois()
		self.idx_components_toss = []
		#rest of properties
		self.cnmf_params = None # CNMF Params: Dict
		self.correlation_img = None #

	def save(self,path):
		'''tmpd = {
			'working_dir':self.working_dir,
			'working_mc_files':self.working_mc_files,
			'working_cnmf_file':self.working_cnmf_file,
			'mc_rig':self.mc_rig,
			'mc_nonrig':self.mc_nonrig,
			'YrDT':self.YrDT,
			'cnmf_results':self.cnmf_results,
			'cnmf_idx_components_keep':self.cnmf_idx_components_keep,
			'cnmf_idx_components_toss':self.cnmf_idx_components_toss,
			'cnmf_params':self.cnmf_params,
			'correlation_img':self.correlation_img
		}'''
		tmp_cnmf_params = {}
		if self.cnmf_params is not None:
			tmp_cnmf_params = self.cnmf_params
			tmp_cnmf_params['dview'] = None  #we cannot pickle socket objects, must set to None
			tmp_cnmf_params['n_processes'] = None

		tmpd = [
			self.working_dir,
			self.working_mc_files,
			self.working_cnmf_file,
			self.mc_mmaps,
			self.mc_rig,
			self.mc_nonrig,
			self.YrDT,
			self.cnmf_results,
			self.idx_components_keep,
			self.idx_components_toss,
			tmp_cnmf_params,
			self.correlation_img,
			self.mc_dsfactors
		]
		save_obj(path, tmpd)
		print("Context saved to: %s" % (path,))

	#@out.capture()()
	def load(self,path,cluster=[]): #cluster is list of: c, dview, n_processes  (from ipyparallel)
		if cluster != []:
			self.c = cluster[0]
			self.dview = cluster[1]
			self.n_processes = cluster[2]

		tmpd = load_obj(path)
		if len(tmpd) == 11: #for backward compatibility
			self.working_dir, self.working_mc_files, self.working_cnmf_file, self.mc_rig, \
			self.mc_nonrig, self.YrDT, self.cnmf_results, self.idx_components_keep, \
			self.idx_components_toss, self.cnmf_params, self.correlation_img = tmpd
		elif len(tmpd) == 12: #for backward compatibility
			self.working_dir, self.working_mc_files, self.working_cnmf_file, self.mc_rig, \
			self.mc_nonrig, self.YrDT, self.cnmf_results, self.idx_components_keep, \
			self.idx_components_toss, self.cnmf_params, self.correlation_img, self.mc_dsfactors = tmpd
		else:
			self.working_dir, self.working_mc_files, self.working_cnmf_file, self.mc_mmaps, self.mc_rig, \
			self.mc_nonrig, self.YrDT, self.cnmf_results, self.idx_components_keep, \
			self.idx_components_toss, self.cnmf_params, self.correlation_img, self.mc_dsfactors = tmpd
		print("Context loaded from: %s" % (path,))
