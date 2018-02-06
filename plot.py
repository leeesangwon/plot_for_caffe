#import pylab as plt
import numpy as np
import matplotlib.pyplot as plt

# FUNC1 : read Result file ########################################################################
def ReadAccAndLoss(trDir, arrOfCurvTr, arrOfCurvVal, displayTerm):

	targetOutTxt = trDir + '/PFP512_VOC++_CVPR.log'
	pFile = open(targetOutTxt, 'r')
	text = pFile.readlines()
	idxLast = len(text) -1
	idxOfStart = -1
	trainOver = False

	if (-1 != text[idxLast].find('Optimization Done.')):
		trainOver = True

	for idx in range(len(text)):
		currTxt = text[idx]
		if -1 != currTxt.find('295] Learning Rate Policy: '):
			idxOfStart = idx
			break

	for idx in range(idxOfStart, idxLast):
		iterationNum = -1
		loss = -1
		valLoss = -1
		valAcc = -1
	
		currTxt = text[idx]

		idxOfTest = -1
		idxOfTest = currTxt.find('433] Iteration')
		if (-1 != idxOfTest) and (-1 != text[idx+1].find('Ignoring source layer')):
			breakIdx = -1
			for idx2 in range(idx, idxLast):
				if -1 != text[idx2].find('detection_eval'):
					breakIdx = idx2
			if (-1 == breakIdx):
				arrOfCurvTr = -1
				arrOfCurvVal = -1
				break
	
		idxOfTr = -1
		idxOfTr1 = currTxt.find('243] Iteration')
		idxOfTr2 = currTxt.find('332] Iteration')
		if -1 != idxOfTr1:
			idxOfTr = idxOfTr1
		elif -1 != idxOfTr2:
			idxOfTr = idxOfTr2
	
		if -1 != idxOfTest:
			idxOfTestEnd = currTxt.find(', Testing net')
			iterationNum = int(currTxt[idxOfTest+15:idxOfTestEnd])

			idxForAcc = 1
			valAccLine = text[idx+idxForAcc]
			if -1==valAccLine.find('detection_eval'):
				while -1==valAccLine.find('detection_eval'):
					idxForAcc +=1 # acc
					#idxForAcc +=2 # avg. recall
					#idxForAcc +=3 # acc IoU
					valAccLine = text[idx+idxForAcc]
			idxOfAcc = valAccLine.find('detection_eval')
			valAcc = float(valAccLine[idxOfAcc+18:])

			#idxForLoss = 1
			#valLossLine = text[idx+idxForLoss]
			#if -1==valLossLine.find('mbox_loss'):
			#	while -1==valLossLine.find('mbox_loss'):
			#		idxForLoss +=1
			#		valLossLine = text[idx+idxForLoss]
			#idxOfLoss = valLossLine.find('mbox_loss')
			#idx2OfLoss = valLossLine.find('(* 1 = ')
			#if (-1!=idxOfLoss & -1!=idx2OfLoss):
			#	valLoss = float(valLossLine[idxOfLoss+11:idx2OfLoss-1])
		elif -1 != idxOfTr:
			idxOfLoss = currTxt.find('(')
			if -1 != idxOfLoss:
				iterationNum = int(currTxt[idxOfTr+15:idxOfLoss-1])
			else:
				idxOfLoss = currTxt.find(', loss = ')
				iterationNum = int(currTxt[idxOfTr+15:idxOfLoss])
			idxOfLoss = currTxt.find(', loss = ')
			loss = float(currTxt[idxOfLoss+9:])
	
		if -1 != iterationNum:
			currIdxIter = iterationNum / displayTerm		
			if -1 != loss:
				#if 0 == len(arrOfCurvTr):
				#	arrOfCurvTr.append([currIdxIter, valLoss, valAcc])
				#elif currIdxIter > arrOfCurvTr[-1][0]:
				#	arrOfCurvTr.append([currIdxIter, loss])
				arrOfCurvTr.append([currIdxIter, loss])
			elif -1 != valAcc:
				if 0 == len(arrOfCurvVal):
					arrOfCurvVal.append([0, 0, 0])
					arrOfCurvVal.append([currIdxIter, valLoss, valAcc])
				elif currIdxIter > arrOfCurvVal[-1][0]:
					arrOfCurvVal.append([currIdxIter, valLoss, valAcc])

	pFile.close()
	return trainOver

# FUNC2 : make array to draw curve ################################################################
def MakeTrCurvArr(arrOfCurvTr):
	trCurv = []
	for idx in range(2):
		trCurv.append( [0]*len(arrOfCurvTr) )
	for idx in range(len(arrOfCurvTr)):
		currArrOfCurvTr = arrOfCurvTr[idx]
		trCurv[0][idx] = currArrOfCurvTr[0]
		trCurv[1][idx] = currArrOfCurvTr[1]
	return trCurv
def MakeValCurvArr(arrOfCurvVal):
	testCurv = []
	for idx in range(3):
		testCurv.append( [0]*len(arrOfCurvVal) )
	for idx in range(len(arrOfCurvVal)):
		currArrOfCurvTest = arrOfCurvVal[idx]
		testCurv[0][idx] = currArrOfCurvTest[0]
		testCurv[1][idx] = currArrOfCurvTest[1]
		testCurv[2][idx] = currArrOfCurvTest[2]
	return testCurv

# FUNC3 : array merge #############################################################################
def MergedTRCurv(outIter, trLoss, curvArrs):
	lastOfIter = 0
	for idx in range(len(curvArrs)):
		currTRarr = MakeTrCurvArr(curvArrs[idx])
		for currArr in currTRarr[0]:
			outIter.append(currArr +lastOfIter)
		trLoss += currTRarr[1]
		if 0 < len(outIter):
			lastOfIter = outIter[-1]

def MergedVALCurv(outIter, valLoss, valACC, curvArrs):
	lastOfIter = 0
	for idx in range(len(curvArrs)):
		currVARarr = MakeValCurvArr(curvArrs[idx])
		for currArr in currVARarr[0]:
			outIter.append(currArr +lastOfIter)
		valLoss += currVARarr[1] 
		valACC += currVARarr[2]
		if 0 < len(outIter):
			lastOfIter = outIter[-1]

# FUNC4 : wrapping function #######################################################################
def GetDataFromDir(outIter, trLoss, valIter, valLoss, valAcc, displayTerm, *Dirs):
	arrTRs = []
	arrVALs = []
	trainOver = False

	for idx in range(len(Dirs)):
		arrTR = []
		arrVAL = []
		trainOver = ReadAccAndLoss(Dirs[idx], arrTR, arrVAL, displayTerm)
		if (-1 != arrTR) and (-1 != arrVAL):
			arrTRs.append(arrTR)
			arrVALs.append(arrVAL)

	MergedTRCurv(outIter, trLoss, arrTRs)
	MergedVALCurv(valIter, valLoss, valAcc, arrVALs)

	return trainOver

# FUNC5 : read solver #############################################################################
def ReadSolverFromDir(Dir):

	targetSoverPrototxt = Dir + '/solver.prototxt'
	fileSolver = open(targetSoverPrototxt, 'r')
	displayTerm = -1
	soverText = fileSolver.readlines()
	for idx in range(len(soverText)):
		soverTxtLine = soverText[idx]
		idxOfDisp = soverTxtLine.find('display')
		if -1 != idxOfDisp:
			displayTerm = int(soverTxtLine[idxOfDisp+9:])
	fileSolver.close()

	return displayTerm

###################################################################################################
# MAIN : make array
###################################################################################################
cOutIter = []
cTrLoss = []
cValIter = []
cValLoss = []
cValAcc = []

compareLegend = 'PFPNet300(VOC07+12):original'
cDir1 = "/home/deep3/caffe/jobs/VGGNet/VOC0712/reference_CVPR"

# Target ###
targetLegend = 'PFPNet512(VOC07++12):Rebuttal,dropout=0.1'
tDir1 = "."
#tDir1 = "/home/deep3/caffe/jobs/VGGNet/VOC0712"

####################################################################################
# plot options
####################################################################################
# MAIN 
legend_loc = 4
#legend_loc = 3
draw_rate = 30 # sec
graph_title = 'PFPNet512(VOC07++12):Rebuttal,dropout=0.1'

# plot compare log
PlotCompare = False
# plot validation loss
PlotVal = False

# plot index (loss)
LossLogScale = False
loss_start = 1
loss_end = 5

# plot index (accuracy)
acc_start = 0.75
acc_end = 0.95

# plot index (# of iters)
iter_start = 0
iter_end = 2400

############################################################################################
# plot
############################################################################################
if True == PlotCompare:
	cDspT = ReadSolverFromDir(cDir1)
	GetDataFromDir(cOutIter, cTrLoss, cValIter, cValLoss, cValAcc, cDspT, cDir1)
fig, ax1 = plt.subplots()
draw11,=ax1.plot(cOutIter, cTrLoss, 'g--')
if True == PlotVal:
	draw31,=ax1.plot(cValIter, cValLoss, 'g--')

tDspT = ReadSolverFromDir(tDir1)
txtOfXLabel = "# of Iters (x %d)" % tDspT
ax1.set_xlabel(txtOfXLabel)
#ax1.set_ylabel('Softmax Loss')
ax1.set_ylabel('Loss')
if True == LossLogScale:
	ax1.set_yscale('log')
plt.ylim(loss_start, loss_end)
plt.grid(True)
bx1 = ax1.twinx()
draw21,=bx1.plot(cValIter, cValAcc, 'r--')
plt.ylim(acc_start, acc_end)
plt.xlim(iter_start, iter_end)
bx1.set_ylabel('Val. Acc.')
plt.title(graph_title, fontsize=23)
plt.grid(True)
firtsPlot = True

# array merge
tOutIter = []
tTrLoss = []
tValIter = []
tValLoss = []
tValAcc = []
trainOver = False

while True:
	tmpOutIter = []
	tmpTrLoss = []
	tmpValIter = []
	tmpValLoss = []
	tmpValAcc = []
	if False == trainOver:
		trainOver = GetDataFromDir(tmpOutIter, tmpTrLoss, tmpValIter, \
					tmpValLoss, tmpValAcc, tDspT, tDir1)
		tOutIter = tmpOutIter
		tTrLoss = tmpTrLoss
		tValIter = tmpValIter
		tValLoss = tmpValLoss
		tValAcc = tmpValAcc
		if True == trainOver:
			draw_rate = 86400
			print('\n Optimization Done.\n')

	draw12,=ax1.plot(tOutIter, tTrLoss, 'b')
	if True == PlotVal:
		draw32,=ax1.plot(tValIter, tValLoss, 'g')
	draw22,=bx1.plot(tValIter, tValAcc, 'r')
	if True == PlotCompare:
		if True == PlotVal:
			plt.legend([draw11, draw12, draw31, draw32, draw21, draw22], \
			['trL-'+compareLegend, 'trL-'+targetLegend, 'vL-'+compareLegend, \
			 'vL-'+targetLegend,'vA-'+compareLegend, 'vA-'+targetLegend], loc=legend_loc)
		else:
			plt.legend([draw11, draw12, draw21, draw22], \
			['trL-'+compareLegend, 'trL-'+targetLegend, 'vA-'+compareLegend, 'vA-'+targetLegend], loc=legend_loc)
	else:
		if True == PlotVal:
			plt.legend([draw12, draw32, draw22], \
			['trL-'+targetLegend, 'vL-'+targetLegend, 'vA-'+targetLegend], loc=legend_loc)
		else:
			plt.legend([draw12, draw22], ['trL-'+targetLegend,'vA-'+targetLegend], \
			loc=legend_loc)
	if True == firtsPlot:
		plt.draw()
		plt.show(block=False)
                plt.subplots_adjust(left=0.05, right=0.92, top=0.97, bottom=0.04)
		firtsPlot = False
	else:
		plt.draw()

	plt.pause(draw_rate)
