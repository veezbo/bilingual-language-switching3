"""
Searchlight script.
run searchlight_example.py
on Ipython.

#Searchlight by Krigeskorte
#http://www.ncbi.nlm.nih.gov/pubmed/16537458
#http://dev.pymvpa.org/examples/searchlight.html

Lines to be modified.

sessionPath = '/home/brain/host/20120808jys-lang/'#Sesson path to be changed
chunksTargets_boldDelay="chunksTargets_boldDelay4-4.txt" #This log file is for boldDelay=4 and stimulusWidth=4

#Also check
		boldDelay=4 #added
		stimulusWidth=4 #added
		
#Change the targets of classification:this script is for Chinese(c) vs Korean(k).
	dataset = dataset[N.array([l in ['c', 'k'] for l in dataset.sa.targets], dtype='bool')]
	print '... and only',dataset.shape[0],'cases of interest (Chinese vs Korean)'

#The default script uses the following anatomical template images. You can change the directory for these templates.
plot_args = {
    'background' : os.path.join(sessionPath,'/home/brain/host/pymvpaniifiles/anat.nii.gz'),
    'background_mask' : os.path.join(sessionPath,'/home/brain/host/pymvpaniifiles/mask_brain.nii.gz'),

#The output file is assigned as follows as an nii image. Change the file name and path.
    niftiresults.to_filename(os.path.join(sessionPath,'analyze/functional/Plang-grey-searchlight.nii'))

#The most important thing is a radius for Searchlight.
        for radius in [3]:
#may be 0,1,2,3.

"""

import mvpa2
import mvpa2.suite as M # machine learning with brain data
import numpy as N
import pylab as P # plotting (matlab-like)
import os # file path processing, directory contents
import datetime # time stamps
import pickle # cPickle
import gzip
import numpy as np
from mvpa2 import cfg
from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.clfs.svm import LinearCSVMC
from mvpa2.measures.base import CrossValidation
from mvpa2.measures.searchlight import sphere_searchlight
from mvpa2.testing.datasets import datasets
from mvpa2.mappers.fx import mean_sample
from mvpa2.datasets.mri import map2nifti
from mvpa2.support.pylab import pl
from mvpa2.misc.plot.lightbox import plot_lightbox

if __debug__:
    M.debug.active += ["SLC"]
numFolds=6

boldDelay=3 #added
stimulusWidth=4 #added

# sessionPath = '/home/brain/host/20120808jys-lang/'
sessionPath = os.getcwd()
preprocessedCache = os.path.join(sessionPath, 'detrendedZscoredMaskedPyMVPAdataset.pkl')
trimmedCache = os.path.join(sessionPath, 'averagedDetrendedZscoredMaskedPyMVPAdataset.pkl')

# LOAD DATASET (Following the instruction of Brian, "and False" is added at lines 29 and 33.)
if os.path.isfile(trimmedCache) and False: 
	print 'loading cached averaged, trimmed, preprocessed dataset',trimmedCache,datetime.datetime.now()
	dataset = pickle.load(gzip.open(trimmedCache, 'rb'))
else:
	if os.path.isfile(preprocessedCache) and False: 
		print 'loading cached preprocessed dataset',preprocessedCache,datetime.datetime.now()
		dataset = pickle.load(gzip.open(preprocessedCache, 'rb', 5))
	else:
		# if not, generate directly, and then cache
		print 'loading and creating dataset',datetime.datetime.now()
		# chunksTargets_boldDelay="chunksTargets_boldDelay4-4.txt" 
		chunksTargets_boldDelay='chunksTargets_boldDelay{0}-{1}-direction.txt'.format(boldDelay, stimulusWidth)
		
		volAttribrutes = M.SampleAttributes(os.path.join(sessionPath,'behavioural',chunksTargets_boldDelay)) 
		dataset = M.fmri_dataset(samples=os.path.join(sessionPath,'analyze/functional/functional4D.nii'),
			targets=volAttribrutes.targets, # I think this was "labels" in versions 0.4.*
			chunks=volAttribrutes.chunks,
			mask=os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr'),
                        add_fa={'vt_thr_glm': os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr')})# added for searchlight

		# DATASET ATTRIBUTES (see AttrDataset)
		print 'functional input has',dataset.a.voxel_dim,'voxels of dimesions',dataset.a.voxel_eldim,'mm'
		print '... or',N.product(dataset.a.voxel_dim),'voxels per volume'
		print 'masked data has',dataset.shape[1],'voxels in each of',dataset.shape[0],'volumes'
		print '... which means that',round(100-100*dataset.shape[1]/N.product(dataset.a.voxel_dim)),'% of the voxels were masked out'
		print 'of',dataset.shape[1],'remaining features ...'
		print 'summary of conditions/volumes\n',datetime.datetime.now()
		print dataset.summary_targets()
		# could add use of removeInvariantFeatures(), but takes a long time, and makes little difference if mask is working well

		# DETREND
		print 'detrending (remove slow drifts in signal, and jumps between runs) ...',datetime.datetime.now() # can be very memory intensive!
		M.poly_detrend(dataset, polyord=1, chunks_attr='chunks') # linear detrend
		print '... done',datetime.datetime.now()

		# ZSCORE
		print 'zscore normalising (give all voxels similar variance) ...',datetime.datetime.now()
		M.zscore(dataset, chunks_attr='chunks', param_est=('targets', ['base'])) # zscoring, on basis of rest periods
		print '... done',datetime.datetime.now()
		print 'saving as compressed file',preprocessedCache


	# AVERAGE OVER MULTIPLE VOLUMES IN A SINGLE TRIAL
	print 'averaging over trials ...',datetime.datetime.now()
	dataset = dataset.get_mapped(M.mean_group_sample(['chunks','targets']))
	print '... only',dataset.shape[0],'cases left now'
	dataset.chunks = N.mod(N.arange(0,dataset.shape[0]),5)

	# REDUCE TO CLASS LABELS, AND ONLY KEEP CONDITIONS OF INTEREST (Japanese VS English)
	dataset.targets = [t[0] for t in dataset.targets]
	# dataset = dataset[N.array([l in ['c', 'k'] for l in dataset.sa.targets], dtype='bool')]
	classificationName='keepswitch'
	dataset = dataset[N.array([l in ['k', 's'] for l in dataset.sa.targets], dtype='bool')]
	print '... and only',dataset.shape[0],'cases of interest (Keep vs Switch)'
	dataset=M.datasets.miscfx.remove_invariant_features(dataset)
	print 'saving as compressed file',trimmedCache
	pickleFile = gzip.open(trimmedCache, 'wb', 5);
	pickle.dump(dataset, pickleFile);

# DO LEARNING AND CLASSIFICATION
# Sometimes Akama added cv=3 temporally.
anovaSelectedSMLR = M.FeatureSelectionClassifier(
	M.PLR(),
	M.SensitivityBasedFeatureSelection(
		M.OneWayAnova(),
		M.FixedNElementTailSelector(500, mode='select', tail='upper')
	),
)
foldwiseCvedAnovaSelectedSMLR = M.CrossValidation(
	anovaSelectedSMLR,
	M.NFoldPartitioner(),
	enable_ca=['samples_error','stats', 'calling_time','confusion']
)

center_ids = dataset.fa.vt_thr_glm.nonzero()[0]

plot_args = {
    'background' : os.path.join(sessionPath,'/home/brain/host/pymvpaniifiles/anat.nii.gz'),
    'background_mask' : os.path.join(sessionPath,'/home/brain/host/pymvpaniifiles/mask_brain.nii.gz'),
    'overlay_mask' : os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr'),
    'do_stretch_colors' : False,
    'cmap_bg' : 'gray',
    'cmap_overlay' : 'autumn', # pl.cm.autumn
    'interactive' : cfg.getboolean('examples', 'interactive', True),
    }

for radius in [3]:
# tell which one we are doing
    print 'Running searchlight with radius: %i ...' % (radius)

    sl = sphere_searchlight(foldwiseCvedAnovaSelectedSMLR, radius=radius, space='voxel_indices',
                            center_ids=center_ids,
                            postproc=mean_sample())

    ds = dataset.copy(deep=False,
                      sa=['targets', 'chunks'],
                      fa=['voxel_indices'],
                      a=['mapper'])

    sl_map = sl(ds)
    sl_map.samples *= -1
    sl_map.samples += 1

    niftiresults = map2nifti(sl_map, imghdr=dataset.a.imghdr)
    niftiresults.to_filename(os.path.join(sessionPath,'analyze/functional/searchlight/{0}-grey-searchlight{1}-{2}.nii'.format(classificationName, boldDelay, stimulusWidth)))
    print 'Best performing sphere error:', np.min(sl_map.samples)

