from itertools import product
import scipy.io
from svmutil import *
import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import sklearn.preprocessing

def combinelists(l1,l2):
	l = [None]*(len(l1)+len(l2))
	l[0:len(l1)] = l1
	l[len(l1):len(l1)+len(l2)] = l2
	return l

def combinealllists(l):
	k=[]

	for i in range(0,len(l)):
		k=combinelists(k,l[i])
	return k

def convertfloor(l):
	for i in range(0,len(l)):
		l[i]=math.floor(l[i])
	return l

def converListInt(l):
	for i in range(0,len(l)):
		l[i]=int(l[i])
	return l

def getcoorFromindex(l1,l2):
	l3=[]
	for i in range(len(l2)):
		l3.append(l1[int(l2[i])])
	return np.asarray(l3)


def convList(l):
	f=[]
	for i in range(0,len(l)):
		f.append(l[i])
	return f

def ToList(l):
	for i in range(0,len(l)):
		l[i]=l[i].tolist()
	return l

def splitdata(full,train_lengths,test_lengths,val_lengths):
	prev=0
	train=[]
	test=[]
	val=[]
	for i in range(0,len(train_lengths)):
		train.append(full[prev:prev+train_lengths[i]])
		prev=prev+train_lengths[i]
		test.append(full[prev:prev+test_lengths[i]])
		prev=prev+test_lengths[i]
		val.append(full[prev:prev+val_lengths[i]])
		prev=prev+val_lengths[i]
		# print train
	return ToList(combinealllists(train)),ToList(combinealllists(test)),ToList(combinealllists(val))

def match(c1,c2):
	count=0
	indices=[]
	for i in range(0,len(c1)):
		if c1[i]==1:
			indices.append(i)
	for j in range(0,len(indices)):
		if c2[indices[j]]==1:
			count=count+1
	return count

def getlengths(t,lengths):
	l=0
	for i in range(0,t):
		l=l+lengths[i]
	return l


def testmodels(fullpca,lengths,m):
	p_acc=[]
	a_val=[]
	a_label=[]
	actual_classes=[]
	# for j in range(0,len(m)):
	for t in range(0,len(lengths)):
		if t==0:
			k1=[]
			k2=[]
			k=[]
			print lengths[0]
			k1=fullpca[0:int(lengths[0])]
			k2=fullpca[(int(lengths[0])):]
			
			k = combinelists(k1,k2)

			classes=[]

			for i in range(0,len(k1)+len(k2)):
				if i<len(k1):
					classes.append(1)
				else:
					classes.append(-1)

			p_label, accuracy, p_val = svm_predict(classes,k, m[t])
			a_val.append(p_val)
			a_label.append(p_label)
			p_acc.append(accuracy)
			actual_classes.append(classes)
			# matchclasses=match(classes,p_label)
			# print matchclasses
			
		else:
			k1=[]
			k2=[]
			k3=[]
			k=[]
			k1=fullpca[int(lengths[t-1]):int(lengths[t]+lengths[t-1])]
			k3=fullpca[int(lengths[t]+lengths[t-1]):]
			k4=fullpca[:int(lengths[t-1])]

			k2=combinelists(k3,k4)
			k=combinelists(k1,k2)
			classes=[]

			for i in range(0,int(getlengths(t,lengths))):
				classes.append(-1)
			for i in range(0,int(lengths[t])):
				classes.append(1)
			for i in range(0,int(len(k)-lengths[t]-getlengths(t,lengths))):
				classes.append(-1)

			p_label, accuracy, p_val = svm_predict(classes, k, m[t])
			a_val.append(p_val)
			a_label.append(p_label)
			p_acc.append(accuracy)
			actual_classes.append(classes)
				# matchclasses=match(classes,p_label)
				# print matchclasses
	return a_label,p_acc,a_val,actual_classes

def generatemodels(fullpca,lengths):
	models=[]

	for t in range(0,len(lengths)):
		if t==0:
			k1=[]
			k2=[]
			k=[]
			print lengths[0]
			k1=fullpca[0:int(lengths[0])]
			k2=fullpca[(int(lengths[0])):]
			
			k = combinelists(k1,k2)

			classes=[]

			for i in range(0,len(k1)+len(k2)):
				if i<len(k1):
					classes.append(1)
				else:
					classes.append(-1)

			m=svm_train(classes,k,'-t 0 -c 5')
			models.append(m)
		else:
			k1=[]
			k2=[]
			k3=[]
			k=[]
			k1=fullpca[int(lengths[t-1]):int(lengths[t]+lengths[t-1])]
			k3=fullpca[int(lengths[t]+lengths[t-1]):]
			k4=fullpca[:int(lengths[t-1])]

			k2=combinelists(k3,k4)
			k=combinelists(k1,k2)

			classes=[]
			for i in range(0,int(getlengths(t,lengths))):
				classes.append(-1)
			for i in range(0,int(lengths[t])):
				classes.append(1)
			for i in range(0,int(len(k)-lengths[t]-getlengths(t,lengths))):
				classes.append(-1)
			# m=svm_train(classes,k,'-t 2 -g 1e-2 -c 50')
			m=svm_train(classes,k,'-t 1 -g 0.5 -c 200	 -d 1')
			# m=svm_train(classes,k,'-t 0 -c 10')

			models.append(m)
	return models


def train(data,lengths,clas,name):
	train_lengths=convertfloor(lengths*.70)
	test_lenghts=convertfloor(lengths*0.15)
	val_lengths=lengths-train_lengths-test_lenghts
	traindata,testdata,valdata=splitdata(data,train_lengths,test_lenghts,val_lengths)
	models=generatemodels(traindata,train_lengths)
	labels,accuracies,values,actual_classes=testmodels(valdata,val_lengths,models)

	conff=[]
	for i in range(0,len(labels)):
		inter=[]
		for j in range(0,len(labels)):
			inter.append(match(actual_classes[j],labels[i]))
		conff.append(inter)
		# print actual_classes[i]
		# print labels[i]
	plotconf(conff,clas,name)
	return accuracy_conf(conff)

def train_without_split(traindata,valdata,train_lengths,val_lengths,clas,name):
	models=generatemodels(traindata,train_lengths)
	labels,accuracies,values,actual_classes=testmodels(valdata,val_lengths,models)

	conff=[]
	for i in range(0,len(labels)):
		inter=[]
		for j in range(0,len(labels)):
			inter.append(match(actual_classes[j],labels[i]))
		conff.append(inter)
		# print actual_classes[i]
		# print labels[i]
	plotconf(conff,clas,name)
	print 'here'
	return accuracy_conf(conff),models,labels,actual_classes

def accuracy_conf(conf):
	total=0
	diag=0.0
	for i in range(0,len(conf)):
		total=total+sum(conf[i])
	for l in range(0,len(conf)):
		for m in range(0,len(conf[l])):
			if m==l:
				diag=diag+conf[l][m]
	print diag
	print total
	return diag/total

def plotconf(conf_arr,alphabet,name):
	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
	                interpolation='nearest')

	width, height = len(conf_arr[0]),len(conf_arr[0])

	for x in xrange(width):
	    for y in xrange(height):
	        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')

	cb = fig.colorbar(res)
	plt.xticks(range(width), alphabet[:width])
	plt.yticks(range(height), alphabet[:height])
	plt.savefig(name, format='png')

def labelsconvert(l):
	for i in range(0,len(l)):
		for j in range(0,len(l[i])):
			if l[i][j]==-1:
				l[i][j]=0
			elif l[i][j]==1:
				l[i][j]=i+1
	return l
def addtwolists(l1,l2):
	for x,y in zip(l1,l2):
		z=x+y
	return z
def remove(k,i):
	k=k.tolist()
	for t in range(0,len(k[0])):
		if k[0][t]==i+1+i:
			k[0][t]=0
	return np.asarray(k)
def addlist(l):
	k=np.zeros((1,len(l[0])),dtype=np.int)	
	for i in range(0,len(l)):
			k=np.asarray(l[i])+k
	return k[0].tolist()

def regionlabels(l):
	k=[None]*len(l[0])
	for i in range(0,len(l)):
		for j in range(0,len(l[i])):
			if l[i][j]==i+1:
				k[j]=i+1

	return k


# fulldata=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fulldata.mat')
# fullpca=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fullpca.mat')
# lengths=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/lengths.mat')

# fulldata=fulldata['fulldata']
# fullpca=fullpca['fullpca']
# lengths=lengths['lengths']

# fulldata=fulldata
# fullpca=fullpca
# lengths=lengths[0]


# print train(fulldata,lengths,'12345','gauss_image')
# print train(fullpca,lengths,'12345','gauss_pca_45_image')

# # dataset 1a question 3
# c1train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class1_train.txt'
# c2train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class2_train.txt'
# c3train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class3_train.txt'
# c4train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class4_train.txt'

# c1val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class1_val.txt'
# c2val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class2_val.txt'
# c3val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class3_val.txt'
# c4val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/linearly_Separable_Data/group16/class4_val.txt'

# trainLlen=[]
# trainL=[]
# valL=[]
# valLlen=[]

# cLtrain=[]
# cLtrain.append(c1train)
# cLtrain.append(c2train)
# cLtrain.append(c3train)
# cLtrain.append(c4train)

# cLval=[]
# cLval.append(c1val)
# cLval.append(c2val)
# cLval.append(c3val)
# cLval.append(c4val)

# for f in cLtrain:
# 	file=open(f,'r')
# 	intermediate=[]
# 	for line in file:
# 	    words = line.split()
# 	    intermediate.append(map(float,words))
# 	trainL.append(intermediate)
# 	trainLlen.append(len(intermediate))

# for f in cLval:
# 	file=open(f,'r')
# 	intermediate=[]
# 	for line in file:
# 	    words = line.split()
# 	    intermediate.append(map(float,words))
# 	valL.append(intermediate)
# 	valLlen.append(len(intermediate))

# trainL=combinealllists(trainL)
# valL=combinealllists(valL)

# print train_without_split(trainL,valL,trainLlen,valLlen,'1234','linear_poly_c_150_g0.5_d2')

# # for dataset 1(b)
# c1train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group16/class1_train.txt'
# c2train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group16/class2_train.txt'

# c1val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group16/class1_val.txt'
# c2val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/non-linearly_Separable/group16/class2_val.txt'

# trainLlen=[]
# trainL=[]
# valL=[]
# valLlen=[]

# cLtrain=[]
# cLtrain.append(c1train)
# cLtrain.append(c2train)

# cLval=[]
# cLval.append(c1val)
# cLval.append(c2val)

# for f in cLtrain:
# 	file=open(f,'r')
# 	intermediate=[]
# 	for line in file:
# 	    words = line.split()
# 	    intermediate.append(map(float,words))
# 	trainL.append(intermediate)
# 	trainLlen.append(len(intermediate))

# for f in cLval:
# 	file=open(f,'r')
# 	intermediate=[]
# 	for line in file:
# 	    words = line.split()
# 	    intermediate.append(map(float,words))
# 	valL.append(intermediate)
# 	valLlen.append(len(intermediate))

# trainL=combinealllists(trainL)
# valL=combinealllists(valL)

# print train_without_split(trainL,valL,trainLlen,valLlen,'12','nonlinear_poly_c_10_g1e-1')

# dataset 1-(c) overlapping dataset
c1train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class1_train.txt'
c2train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class2_train.txt'
c3train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class3_train.txt'

c1val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class1_val.txt'
c2val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class2_val.txt'
c3val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class3_val.txt'

trainLlen=[]
trainL=[]
valL=[]
valLlen=[]

cLtrain=[]
cLtrain.append(c1train)
cLtrain.append(c2train)
cLtrain.append(c3train)

cLval=[]
cLval.append(c1val)
cLval.append(c2val)
cLval.append(c3val)

for f in cLtrain:
	file=open(f,'r')
	intermediate=[]
	for line in file:
	    words = line.split()
	    intermediate.append(map(float,words))
	trainL.append(intermediate)
	trainLlen.append(len(intermediate))

for f in cLval:
	file=open(f,'r')
	intermediate=[]
	for line in file:
	    words = line.split()
	    intermediate.append(map(float,words))
	valL.append(intermediate)
	valLlen.append(len(intermediate))

trainL=combinealllists(trainL)
valL=combinealllists(valL)

# trainL=sklearn.preprocessing.normalize(np.asarray(trainL)).tolist()
# valL=sklearn.preprocessing.normalize(np.asarray(valL)).tolist()

accuracy,models,labels,actual_classes= train_without_split(trainL,valL,trainLlen,valLlen,'123','overlapping_poly_c_80_g0.5_p_3')
print accuracy

complete_labels=labelsconvert(labels)
complete_class=labelsconvert(actual_classes)
added_class=addlist(complete_class)

valL=np.asarray(valL)
# Plotting decision regions
x_min, x_max = valL[:, 0].min() - 1, valL[:, 0].max() + 1
y_min, y_max = valL[:, 1].min() - 1, valL[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

meshpoints=(np.c_[xx.ravel(), yy.ravel()]).tolist()
lol=[]
for i in range(0,len(models)):
	if i<len(models)-1:
		lol.append(len(meshpoints)/len(models))
	else:
		lol.append(len(meshpoints)-2*len(meshpoints)/len(models))


labels,accuracies,values,actual_classes=testmodels(meshpoints,lol,models)
labels=labelsconvert(labels)

labels=regionlabels(labels)
labels=np.asarray(labels).reshape(xx.shape)



# one=np.asarray(labels[0]).reshape(xx.shape)
# two=np.asarray(labels[1]).reshape(xx.shape)
# three=np.asarray(labels[2]).reshape(xx.shape)

# plt.contourf(xx,yy,one)
# plt.contourf(xx,yy,two)
# plt.contourf(xx,yy,three)



# Z=np.asarray(addlist(labels))
# Z.reshape(xx.shape())

# f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))
# # values=np.asarray(values[0])
# # Z=
# # Z=getcoorFromindex(trainL,models[0].get_sv_indices())
# # axarr[0,0].contourf(xx,yy,Z, alpha=0.4)
# axarr[0,0].scatter(valL[:, 0], valL[:, 1], c=complete_class, alpha=0.8)
# axarr[0,0].set_title('first one')

# plt.show()

