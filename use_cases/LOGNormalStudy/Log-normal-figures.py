#%% activity MITYA
import glob
import numpy as np
import pylab as pl
from scipy.signal import savgol_filter

ffllss = glob.glob('/mnt/ceph/neuro/DataForPublications/log-normal-data/*_act.npz')
ffllss.sort()
pl.figure("Distribution of population activity (fluorescence)")
print(ffllss)
for thresh in np.arange(1.5,4.5,1):
    count = 1
    pl.pause(0.1)
    for ffll in ffllss:

        with np.load(ffll) as ld:
            print(ld.keys())
            locals().update(ld)
            pl.subplot(3,4,count)
            count += 1
            a = np.histogram(np.sum(comp_SNR_trace>thresh,0)/len(comp_SNR_trace),100)
            pl.plot(np.log10(a[1][1:][a[0]>0]),savgol_filter(a[0][a[0]>0]/comp_SNR_trace.shape[-1],5,3))

            pl.title(ffll.split('/')[-1])

pl.legend(np.arange(1.5,4.5,1))
pl.xlabel('fraction active')
pl.ylabel('fraction of frames')
#%% remove outliers
def remove_outliers(data, num_iqr = 5):
    median = np.median(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25

    min_ = q25 - (iqr*num_iqr)
    max_ = q75 + (iqr*num_iqr)
    new_data = data[(data>min_) & (data<max_)]
    return new_data
#%%
from caiman.components_evaluation import mode_robust
from scipy.stats import norm # quantile function
def estimate_stats(log_values, n_bins = 30, delta = 3 ):
    a = np.histogram(log_values,bins=n_bins)
    mu_est = np.argmax(a[0])
    bins = a[1][1:]
    bin_size = np.diff(a[1][1:])[0]
    pdf = a[0]/a[0].sum()/bin_size
    # compute the area around the mean and from that infer the standard deviation (see  pic on phone June 7 2018)
    area_PDF = np.sum(np.diff(bins[mu_est-delta-1:mu_est+delta])*pdf[mu_est-delta:mu_est+delta])
    alpha = delta*bin_size
    sigma = alpha / norm.ppf( (area_PDF+1)/2 )
    mean = bins[mu_est]-bin_size/2
    return bins, pdf, mean, sigma
#%% spikes MITYA population synchrony
import glob
import numpy as np
import pylab as pl
from scipy.signal import savgol_filter
from sklearn.preprocessing import normalize

# trying to recover this data. If I will be able to I will move it to /mnt/ceph/neuro/DataForPublications/log-normal-data/
ffllss = glob.glob('/mnt/home/agiovann/SOFTWARE/CaImAn/use_cases/CaImAnpaper/*_spikes_DFF.npz')
#        mode = 'inter_spike_interval'
mode = 'synchrony' # 'synchrony', 'firing_rates'
mode = 'firing_rates' # 'synchrony', 'firing_rates'
mode = 'correlation'
pl.figure("Distribution of " + mode + " (spikes)", figsize=(10,10))

ffllss.sort()
print(ffllss)
for thresh in [0]:
    count = 1
    for ffll in ffllss[:]:
        with np.load(ffll) as ld:
            print(ffll.split('/')[-1])
            print(ld['S'].shape)
            locals().update(ld)


            if mode == 'synchrony':
                pl.subplot(3,4,count)
                S = np.maximum(S,0)
                if True:
                    S /= np.max(S)
                else:
                    S /= np.max(S,1)[:,None] # normalized to maximum activity of each neuron
                S[np.isnan(S)] = 0
                activ_neur_fraction = np.sum(S,0)
                activ_neur_fraction = remove_outliers(activ_neur_fraction)
                activ_neur_fraction = np.delete(activ_neur_fraction,np.where(activ_neur_fraction<=1e-4)[0])
                activ_neur_fraction_log =  np.log10(activ_neur_fraction )
                bins, pdf, mean, sigma = estimate_stats(activ_neur_fraction_log, n_bins = 30, delta = 3 )
                pl.plot(bins,norm.pdf(bins,loc = mean, scale = sigma))
                pl.plot(bins,pdf,'-.')
                pl.legend(['fit','data'])
                pl.xlabel('fraction of max firing')
                pl.ylabel('probability')
            elif mode == 'firing_rates':
                pl.subplot(3,4,count)
                fir_rate = np.mean(S,1)
                fir_rate = remove_outliers(fir_rate, num_iqr = 30)
                fir_rate = np.delete(fir_rate,np.where(fir_rate==0)[0])
                fir_rate_log = np.log10(fir_rate)
                bins, pdf, mean, sigma = estimate_stats(fir_rate_log, n_bins = 20, delta = 3)
                pl.plot(bins,norm.pdf(bins,loc = mean, scale = sigma))
                pl.plot(bins,pdf,'-.')
                pl.legend(['fit','data'])
                pl.xlabel('firing rate')
                pl.ylabel('number of neurons')
            elif mode == 'correlation':
                pl.subplot(3,4,count)
                S = normalize(S,axis=1)
                cc = S.dot(S.T)
                cc = cc[np.triu_indices(cc.shape[0], k = 1)]
                cc[np.isnan(cc)] = 0
                cc = np.delete(cc, np.where(cc<1e-5)[0])
                cc_log = np.log10(cc)
                bins, pdf, mean, sigma = estimate_stats(cc_log, n_bins = 30, delta = 3)
                pl.plot(bins,norm.pdf(bins,loc = mean, scale = sigma))
                pl.plot(bins,pdf,'-.')
                pl.legend(['fit','data'])
                pl.xlabel('corr. coeff.')
                pl.ylabel('count')

            count += 1
#                    pl.plot(np.log10(a[1][1:][a[0]>0]),savgol_filter(a[0][a[0]>0]/S.shape[-1],5,3))

            pl.title(ffll.split('/')[-1][:18])
            pl.pause(0.1)

pl.tight_layout(w_pad = 0.05)
pl.savefig('/mnt/xfs1/home/agiovann/ceph/LinuxDropbox/Dropbox (Simons Foundation)/Lab Meetings & Pres/MITYA JUNE 2018/'+mode+'.pdf')

#%%

isis = ([np.histogram(np.log10(np.diff(np.where(s>0)[0]))) for s in S])
for isi in isis[::20]:
    pl.plot((isi[1][1:][isi[0]>0]),savgol_filter(isi[0][isi[0]>0]/S.shape[-1],5,3))