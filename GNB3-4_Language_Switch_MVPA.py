"""
global accuracy

You might want to save the log on the shell to a text file.

Lines to be modified if necessary.

sessionPath = '/home/brain/host/20120808jys-lang/'#Sesson path to be changed
                chunksTargets_boldDelay="chunksTargets_boldDelay4-4.txt" #This log file is for boldDelay=4 and stimulusWidth=4

#Also check
		boldDelay=4 #added
		stimulusWidth=4 #added
		
#Change if you need the targets of classification:this script is for Japanese(j) vs English(e).
	dataset = dataset[N.array([l in ['m', 't'] for l in dataset.sa.targets], dtype='bool')]
"""
import mvpa2.suite as M
import numpy as N
import pylab as P # plotting (matlab-like)
import os # file path processing, directory contents
import datetime # time stamps
import pickle # cPickle
import gzip

numFolds=6

# sessionPath = '/home/brain/host/20130908ln'
sessionPath = os.getcwd()
preprocessedCache = os.path.join(sessionPath, 'detrendedZscoredMaskedPyMVPAdataset.pkl')
trimmedCache = os.path.join(sessionPath, 'averagedDetrendedZscoredMaskedPyMVPAdataset.pkl')

boldDelay=3 #added
stimulusWidth=4 #added

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
		# chunksTargets_boldDelay="chunksTargets_boldDelay4-4.txt" #Modified
		chunksTargets_boldDelay="chunksTargets_boldDelay{0}-{1}-LanguageSwitch-Japanese_English.txt".format(boldDelay, stimulusWidth)
		
		volAttribrutes = M.SampleAttributes(os.path.join(sessionPath,'behavioural',chunksTargets_boldDelay)) # default is 3.txt.
		# print volAttribrutes.targets
		# print len(volAttribrutes.targets)
		# print volAttribrutes.chunks
		# print len(volAttribrutes.chunks)
		dataset = M.fmri_dataset(samples=os.path.join(sessionPath,'analyze/functional/functional4D.nii'),
			targets=volAttribrutes.targets, # I think this was "labels" in versions 0.4.*
			chunks=volAttribrutes.chunks,
			mask=os.path.join(sessionPath,'analyze/structural/lc2ms_deskulled.hdr'))

		# DATASET ATTRIBUTES (see AttrDataset)
		print 'functional input has',dataset.a.voxel_dim,'voxels of dimesions',dataset.a.voxel_eldim,'mm'
		print '... or',N.product(dataset.a.voxel_dim),'voxels per volume'
		print 'masked data has',dataset.shape[1],'voxels in each of',dataset.shape[0],'volumes'
		print '... which means that',round(100-100*dataset.shape[1]/N.product(dataset.a.voxel_dim)),'% of the voxels were masked out'
		print 'of',dataset.shape[1],'remaining features ...'
		print 'summary of conditions/volumes\n',datetime.datetime.now()
		print dataset.summary_targets()

		# DETREND
		print 'detrending (remove slow drifts in signal, and jumps between runs) ...',datetime.datetime.now() # can be very memory intensive!
		M.poly_detrend(dataset, polyord=1, chunks_attr='chunks') # linear detrend
		print '... done',datetime.datetime.now()

		# ZSCORE
		print 'zscore normalising (give all voxels similar variance) ...',datetime.datetime.now()
		M.zscore(dataset, chunks_attr='chunks', param_est=('targets', ['base'])) # zscoring, on basis of rest periods
		print '... done',datetime.datetime.now()
		#P.savefig(os.path.join(sessionPath,'pyMVPAimportDetrendZscore.png'))

		pickleFile = gzip.open(preprocessedCache, 'wb', 5);
		pickle.dump(dataset, pickleFile);

	# AVERAGE OVER MULTIPLE VOLUMES IN A SINGLE TRIAL
	print 'averaging over trials ...',datetime.datetime.now()
	dataset = dataset.get_mapped(M.mean_group_sample(attrs=['chunks','targets']))
	print '... only',dataset.shape[0],'cases left now'
	dataset.chunks = N.mod(N.arange(0,dataset.shape[0]),5)

	# print '\n\n\n'
	# print dataset.targets
	# print len(dataset.targets)
	# print dataset.chunks
	# print len(dataset.chunks)

	# REDUCE TO CLASS LABELS, AND ONLY KEEP CONDITIONS OF INTEREST (JAPANESE VS ENGLISH)
	dataset.targets = [t[0:2] for t in dataset.targets]
	dataset = dataset[N.array([l in ['jj', 'je', 'ej', 'ee'] for l in dataset.sa.targets], dtype='bool')]
	print '... and only',dataset.shape[0],'cases of interest (Language Switch between Japanese vs English)'
	dataset=M.datasets.miscfx.remove_invariant_features(dataset)
	print 'saving as compressed file',trimmedCache
	pickleFile = gzip.open(trimmedCache, 'wb', 5);
	pickle.dump(dataset, pickleFile);


anovaSelectedSMLR = M.FeatureSelectionClassifier(
	M.GNB(common_variance=True),
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
# run classifier
print 'learning on detrended, normalised, averaged, Language Switch ...',datetime.datetime.now()
results = foldwiseCvedAnovaSelectedSMLR(dataset)
print '... done',datetime.datetime.now()
print 'accuracy',N.round(100-N.mean(results)*100,1),'%',datetime.datetime.now()

#New lines for out putting the result into a csv file.
precision=N.round(100-N.mean(results)*100,1)
st=str(boldDelay) + ',' + str(stimulusWidth) + ',' + str(precision) +'\n'
f = open( "withinPredictionResult.csv", "a" )
f.write(st)
f.close


# display results
P.figure()
P.title(str(N.round(foldwiseCvedAnovaSelectedSMLR.ca.stats.stats['ACC%'], 1))+'%, n-fold SMLR with anova FS x 500')
foldwiseCvedAnovaSelectedSMLR.ca.stats.plot()
P.savefig(os.path.join(sessionPath,'confMatrixAvTrial{0}-{1}-LanguageSwitch-Japanese_English-GNB.png'.format(boldDelay, stimulusWidth)))
print foldwiseCvedAnovaSelectedSMLR.ca.stats.matrix

print 'accuracy',N.round(foldwiseCvedAnovaSelectedSMLR.ca.stats.stats['ACC%'], 1),'%',datetime.datetime.now()

# this should give average anova measure over the folds - but in fact would be much the same as taking over single fold
sensana = anovaSelectedSMLR.get_sensitivity_analyzer(postproc=M.maxofabs_sample())
cv_sensana = M.RepeatedMeasure(sensana, M.NFoldPartitioner())
sens = cv_sensana(dataset)
print sens.shape
M.map2nifti(dataset, N.mean(sens,0)).to_filename("anovaSensitivity{0}-{1}-LanguageSwitch-Japanese-English-GNB.nii".format(boldDelay, stimulusWidth))

# this looks good, but don't know way to get back from this feature selected space (of 500) to the whole space of 28k or so, for output
weights = anovaSelectedSMLR.clf.weights 






