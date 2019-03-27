t_online = []
t_detect = []
t_shapes = []
for fls in glob.glob('*_NEW*.npz'):
    with np.load(fls) as ld:
        print(ld.keys())
        t_online.append(ld['t_online'])
        t_detect.append(ld['t_detect'])
        t_shapes.append(ld['t_shapes'])


pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}
pl.rc('font', **font)

pl.figure()
pl.stackplot(np.arange(len(t_detect.max(0))), 1e3 * (np.array(t_online.max(0)) - np.array(t_detect.max(0)) - np.array(t_shapes.max(0))),
              1e3 * np.array(t_detect.max(0)), 1e3 * np.array(t_shapes.max(0)))
pl.title('Processing time per frame')
pl.xlabel('Frame #')
pl.ylabel('Processing time [ms]')
pl.ylim([0, 5000])
pl.legend(labels=['process', 'detect', 'shapes'])