import os # paths, directory contents
import re # string matching
import codecs # reading encoded files (see notes at end)
import sys

### HOW TO USE ###
#
# This script will read in Eprime log files (in their original text format, not the derived .edat format.), and output a PyMVPA file specifying the experimental conditions/activity
# The output is the PyMVPA standard format of two columns: condition label [tab] run number (in PyMVPA parlance these are called "targets" and "chunks" respectively)
#
# To use it you have to:
#	- define the base directory for the participant
#	- the program assumes that from there $base/behavioural/*.txt is the eprime log file. If not, specify the file by hand. If there were more than one session (e.g. the experiment was interupted), you should combine the multiples logs into a single log, e.g. with the linux command 'cat'.
#	- make sure the category markers, for each condition, are unique, also with respect to the rest ('r') and baseline ('b') conditions
#	- fix the chunk/run interval to the number of trials there were in each run
#	- you can output the volume labels with or without a Bold delay applied
#		- without the bold delay, make boldDelayVols=0, and the stimulusVols should be exactly equal to that in the presentation. If you do this, you will have to apply the delay from within PyMVPA
#		- with the bold delay, make boldDelay=X, and stimulusVols a bit longer than in reality, since it gets smudged out in time
#	
#

#sessionPath = sys.argv[1]
# sessionPath = '/home/brain/host/20110301'
sessionPath = os.getcwd()
behavFilename = [f for f in os.listdir(os.path.join(sessionPath,'behavioural')) if f.endswith('-log.txt')][0] # guess that the first text file in the behavioural directory is the ePrime output

# single character markers for three stimulus category conditions
categoryMarkers = { 'Japanese': 'j',	
	'English': 'e'}
categoryName = "Language" + ": "
categoryParameter = "English: "

trialsPerRun = 40 # number of trials in each run

trialVols = 10 # number of volumes per stimulus trials
stimulusVols = 4 # number of volumes containing stimulus-specific activity (including BOLD smudging, if relevant; remaining trial volumes assumed to be rest):default=4
boldDelayVols = 3 # BOLD delay to apply :default=3
runInVols = 0 # number of baseline volumes at start of scanning run (= a PyMVPA chunk)
runOutVols = 40 # number of baseline volumes at end of scanning run
	 
if boldDelayVols>-1:
	pymvpaFileName = 'chunksTargets_boldDelay'+str(boldDelayVols)+'-'+str(stimulusVols)+'-'+categoryMarkers.keys()[0]+'_'+categoryMarkers.keys()[1]+'.txt'
else:
	pymvpaFileName = 'chunksTargets.txt'

### PROGRAM OUTLINE ###
# read text file line by line
	# at start of each run
		# if chunk >-1, output 15 x "base chunk" ### for bold delay, add 3 here ...
		# increment counter
		# output 15 x "base chunk" (or 16 in original experiments) 
	# at start of each trial, increment trial counter
	# find category and stimulus name
		# output 3 x "m/t/o-name chunk"
		# output 7 x "rest chunk"
# output 15 x "base chunk", and subtract 3 here?

trialCounter = 0; # keeps track of number of stimuli, for knowing where to put in run-breaks
chunkCounter = 0; # chunk (run) counter - used by PyMVPA for detrending, and for cross-validation partitions
concept = ''
category =''
#print 'opening behavioural log file',behavFilename
behavFile = codecs.open(os.path.join(sessionPath,'behavioural',behavFilename), 'r', 'utf_16') # actually 'utf_16' is enough, don't need 'utf_16_le'
pymvpaFile = open(os.path.join(sessionPath,'behavioural',pymvpaFileName),'w')

print "input file:",os.path.join(sessionPath,'behavioural',behavFilename)


for line in behavFile:
	#print line
	if re.search(categoryName,line):
		category=categoryMarkers[re.search(categoryName+'(.+)\r',line).groups(0)[0]] # REMEMBER - to choose \r or \n, depending on Op. System used to produce file
		# print concept, line
	if re.search(categoryParameter,line):
		concept=re.search(categoryParameter+'(.+)\n',line).groups(0)[0].strip()
		# print category, line
		trialCounter += 1
		if trialCounter % trialsPerRun == 1:
			if chunkCounter > 0:
				#print "CHUNK END>", chunkCounter
				pymvpaFile.write(("base %i\n" % chunkCounter) * (runOutVols-boldDelayVols)) # e.g. 15s-3s
			chunkCounter += 1;
			#print "<CHUNK START", chunkCounter
			pymvpaFile.write(("base %i\n" % chunkCounter) * (runInVols+boldDelayVols)) # e.g. 15s+3s
		#print "\t[%i] %s-%s %i" % (trialCounter, category, concept, chunkCounter) 
		if category=="j":
			pymvpaFile.write(("%s-j%s %i\n" % (category, concept, chunkCounter)) * stimulusVols) # e.g. 3s
		else:
			pymvpaFile.write(("%s-%s %i\n" % (category, concept, chunkCounter)) * stimulusVols) # e.g. 3s

		pymvpaFile.write(("rest %i\n" % (chunkCounter)) * (trialVols-stimulusVols)) # e.g. 7s
#print "CHUNK END>", chunkCounter
pymvpaFile.write(("base %i\n" % chunkCounter) * (runOutVols-boldDelayVols)) # e.g. 15s-3s
pymvpaFile.write(("base %i\n" % chunkCounter)) # extra one to account for accidental attachment of mean image to end of nifti sequence
		
print "output chunk/target file with total %d trials in %d chunks (double-check: remainder should be zero: %d)" % (trialCounter, chunkCounter, trialCounter % trialsPerRun)
print "output file:",os.path.join(sessionPath,'behavioural',pymvpaFileName)


# base 0
# m-zebra 0
# rest 0
# ...
# t-hammer 1
# ...
# o-hail 7


# ENCODING PROBLEM:
#
# seems to be UTF 16 (UCS2) Little-Endian encoded according to python codecs idea of BOM
#
#	In [122]: behavFile = open('/home/brian/workingData/19820303HAGO_201009280800/behavioural/old/MitchellRepImageOrthoChinese-2-1.txt', 'r')
#	In [123]:
### HOW TO USE ### line = behavFile.next()
#	In [124]: line.startswith(codecs.BOM_UTF16_BE)
#	Out[124]: False
#	In [125]: line.startswith(codecs.BOM_UTF16_LE)
#	Out[125]: True
#
# and according to wiki page: http://en.wikipedia.org/wiki/UTF-16/UCS-2
# "This results in the byte sequence 0xFE,0xFF for big-endian, or 0xFF,0xFE for little-endian."
#
#	brian@brian2-desktop:~/workingData/19820303HAGO_201009280800/behavioural$ od -c -t u1 -t x1 old/MitchellRepImageOrthoChinese-2-1.txt  | more
#	0000000 377 376   *  \0   *  \0   *  \0      \0   H  \0   e  \0   a  \0
#		    255 254  42   0  42   0  42   0  32   0  72   0 101   0  97   0
#		     ff  fe  2a  00  2a  00  2a  00  20  00  48  00  65  00  61  00
#	0000020   d  \0   e  \0   r  \0      \0   S  \0   t  \0   a  \0   r  \0
#		    100   0 101   0 114   0  32   0  83   0 116   0  97   0 114   0
#		     64  00  65  00  72  00  20  00  53  00  74  00  61  00  72  00
#
#	... so use codecs.open like this:
# behavFile = codecs.open('/home/brian/workingData/19820303HAGO_201009280800/behavioural/old/MitchellRepImageOrthoChinese-2-1.txt', 'r', 'utf_16_le')
#	... but get luan-ma





