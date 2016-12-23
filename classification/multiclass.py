from itertools import product
import scipy.io
from svmutil import *
import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir
from os.path import join
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.mixture as mixture
import tensorflow as tf 
from tensorflow.contrib import learn
import random
import pandas as pd

def print_confusion_matrix(plabels,tlabels):
	plabels = pd.Series(plabels)
	tlabels = pd.Series(tlabels)

	# draw a cross tabulation...
	df_confusion = pd.crosstab(tlabels,plabels, rownames=['Actual'], colnames=['Predicted'], margins=True)

	#print df_confusion
	return df_confusion

def confusionMatrix(text,Labels,y_pred, not_partial):
    y_actu = np.where(Labels[:]==1)[1]
    df = print_confusion_matrix(y_pred,y_actu)
    print "\n",df
    #print plt.imshow(df.as_matrix())
    if not_partial:
       print "\n",classification_report(y_actu, y_pred)
    print "\n\t------------------------------------------------------\n"

def do_eval(message, sess, correct_prediction, accuracy, pred, X_, y_,x,y):
    predictions = sess.run([correct_prediction], feed_dict={x: X_, y: y_})
    prediction  = tf.argmax(pred,1)
    labels = prediction.eval(feed_dict={x: X_, y: y_}, session=sess)
    print message, accuracy.eval({x: X_, y: y_}),"\n"
    confusionMatrix("Partial Confusion matrix",y_,predictions[0], False)#Partial confusion Matrix
    confusionMatrix("Complete Confusion matrix",y_,labels, True) #complete confusion Matrix

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

def getcoorFromindex(l1,l2,l4,c):
	l3=[]
	l5=[]
	for i in range(0,len(l2)):
		if int(l4[i][0])==c or int(l4[i][0])==-c:
			# print l2[i]
			l3.append(l1[int(l2[i]-1)])
		else:
			l5.append(l1[int(l2[i])-1])
			# print l2[i]

	return np.asarray(l3),np.asarray(l5)

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
	plt.close('all')

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

def getclasses(lengths):
	classes=[]
	for i in range(0,len(lengths)):
		for j in range(0,lengths[i]):
			classes.append(i+1)
	return classes

def pick(l,indices):
	p=[]
	for i in range(0,len(indices)):
		p.append(l[indices[i]])
	return p

def converbatchdim(batch):
	dim=batch.shape[1]
	for i in range(0,batch.shape[0]):
		batch[i].reshape(1,dim)
	return batch

def convertLatent(batchLabel):
	batchLabel=np.asarray(batchLabel)
	batchLabel=batchLabel.tolist()
	f=[]
	for i in range(len(batchLabel)):
		y=np.zeros(max(batchLabel)+1)
		y[batchLabel[i]]=1
		f.append(y.tolist())
	return np.asarray(f,dtype=float)
# # question 3
# c1train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class1_train.txt'
# c2train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class2_train.txt'
# c3train='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class3_train.txt'

# c1val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class1_val.txt'
# c2val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class2_val.txt'
# c3val='/home/ajwahir/acads/pr/assignment/Dataset_Assignment1/Dataset-1_2Dimensional/overlapping_data/group16/class3_val.txt'

# trainLlen=[]
# trainL=[]
# valL=[]
# valLlen=[]

# cLtrain=[]
# cLtrain.append(c1train)
# cLtrain.append(c2train)
# cLtrain.append(c3train)

# cLval=[]
# cLval.append(c1val)
# cLval.append(c2val)
# cLval.append(c3val)

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


# This is for the image data

fulldata=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fulldata.mat')
fullpca=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fullpca.mat')
lengths=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/lengths.mat')
fullpca_20=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fullpca_20.mat')
fullpca_25=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fullpca_25.mat')
fullpca_31=scipy.io.loadmat('/home/ajwahir/acads/pr/assignemt2/data/fullpca_31.mat')



fulldata=fulldata['fulldata']
fullpca=fullpca['fullpca']
lengths=lengths['lengths']
fulldata=fullpca_20['fullpca_20']
# fulldata=fullpca_25['fullpca_25']
# fulldata=fullpca_31['fullpca_31']


lengths=lengths[0]

train_lengths=np.asarray(convertfloor(lengths*.70),dtype='int')
test_lenghts=np.asarray(convertfloor(lengths*0.15),dtype='int')
val_lengths=lengths-train_lengths-test_lenghts
traindata,testdata,valdata=splitdata(fulldata,train_lengths,test_lenghts,val_lengths)




valL=np.asarray(valdata,dtype=np.float32)
trainL=np.asarray(traindata,dtype=np.float32)
trainLlen=train_lengths
valLlen=val_lengths

h1=10
h2=5

classtrain=getclasses(trainLlen)
classval=getclasses(valLlen)

classtrain=np.asarray(classtrain)-1
classval=np.asarray(classval)-1

val_classes=classval

x = tf.placeholder(tf.float32, [None, len(trainL[0])])
sess = tf.InteractiveSession()
y_ = tf.placeholder(tf.float32, shape=[None, max(classtrain)+1])

# W1 = tf.Variable(tf.zeros([len(trainL[0]),h1]))
# b1 = tf.Variable(tf.zeros([h1]))

W1 = tf.Variable(tf.zeros([len(trainL[0]),max(classtrain)+1]))
b1 = tf.Variable(tf.zeros([max(classtrain)+1]))

# W2 = tf.Variable(tf.zeros([h1,max(classtrain)+1]))
# b2 = tf.Variable(tf.zeros([max(classtrain)+1]))

# W2 = tf.Variable(tf.zeros([h1,h2]))
# b2 = tf.Variable(tf.zeros([h2]))

# W3 = tf.Variable(tf.zeros([h2,max(classtrain)+1]))
# b3 = tf.Variable(tf.zeros([max(classtrain)+1]))

sess.run(tf.global_variables_initializer())

y1 = tf.matmul(x,W1) + b1
# y2 = tf.nn.relu(tf.matmul(y1,W2) + b2)
# y3 = tf.matmul(y2,W3) + b3

# W = tf.Variable(tf.zeros([len(trainL[0]),max(classtrain)+1]))
# b = tf.Variable(tf.zeros([max(classtrain)+1]))

# sess.run(tf.global_variables_initializer())

# y=tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y1, y_))
orderedindices=[]
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for i in range(0,len(trainL)):
	orderedindices.append(i)

for i in range(1000):
	random.shuffle(orderedindices)
	indices=orderedindices[0:100]
	batch = np.asarray(pick(trainL,indices))
	batchLabel=np.asarray(pick(classtrain,indices))
	# batch=batch.reshape(100,1,batch.shape[1])
	batchLabel=convertLatent(batchLabel)
	# batchLabel=batchLabel.reshape(100,1,batchLabel.shape[1])
	train_step.run(feed_dict={x: batch, y_: batchLabel})


correct_prediction = tf.equal(tf.argmax(y1,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
valL=np.asarray(valL,dtype=float)
classval=convertLatent(classval)
y_true = np.argmax(classval,1)
val_accuracy, y_pred = sess.run([accuracy, y1], feed_dict={x:valL, y_:classval})
prediction  = tf.argmax(y1,1)
labels = prediction.eval(feed_dict={x: valL, y_: classval}, session=sess)
conff=confusion_matrix(val_classes,labels)

print "Accuracy is " 
print val_accuracy
print "The confusion matrix " 
print conff
# do_eval("Accuracy of Gold Test set Results: ", sess, correct_prediction, accuracy, y1, valL, classval, x, y_)

# print(accuracy.eval(feed_dict={x: valL, y_:classval}))
# classifier = learn.DNNClassifier(hidden_units=[10], n_classes=5, feature_columns=learn.infer_real_valued_columns_from_input(np.asarray(trainL)), optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05))

# classifier.fit(np.asarray(trainL), np.asarray(classtrain), batch_size=128, steps=100)
# cnf_matrix = confusion_matrix(classifier.predict(np.asarray(valL)), classval)

# gmm = mixture.GaussianMixture(n_components=3, covariance_type='full',reg_covar=1e-06).fit(np.asarray(trainL))
# pred=gmm.predict(valL)




# c=50

# # params='-t 2 -g 1e-2 -c 50'	
# params='-t 1 -g 0.020833333333333332 -c 200 -d 3'
# # params='-t 0 -c 5 ' 
# clss='12345'

# m=svm_train(classtrain,trainL,params)
# p_label, accuracy, p_val = svm_predict(classval,valL, m)
# cnf_matrix = confusion_matrix(classval, p_label)
# name=params+'  ' + `accuracy[0]`+' '+ clss
# print cnf_matrix
# plotconf(cnf_matrix.tolist(),clss,params+'  ' + `accuracy[0]`)


# valL=np.asarray(valL)
# # Plotting decision regions
# x_min, x_max = valL[:, 0].min() - 1, valL[:, 0].max() + 1
# y_min, y_max = valL[:, 1].min() - 1, valL[:, 1].max() + 1
# # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
# #                      np.arange(y_min, y_max, 0.1))

# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))
# meshpoints=(np.c_[xx.ravel(), yy.ravel()]).tolist()

# lol=[]
# for i in range(0,len(valLlen)):
# 	if i<len(valLlen)-1:
# 		lol.append(len(meshpoints)/len(valLlen))
# 	else:
# 		lol.append(len(meshpoints)-(len(valLlen)-1)*len(meshpoints)/len(valLlen))
# lol=getclasses(lol)

# SVb,SVu=getcoorFromindex(trainL,m.get_sv_indices(),m.get_sv_coef(),c)


# Z, acc, p_val = svm_predict(lol,meshpoints, m)
# Z=np.asarray(Z)
# Z=Z.reshape(xx.shape)
# plt.contourf(xx,yy,Z, alpha=0.4)
# plt.scatter(valL[:, 0], valL[:, 1], c=classval, alpha=0.8)
# # plt.savefig(name+' decision ', format='png')

# # plt.close('all')
# # plt.contourf(xx,yy,Z, alpha=0.4)
# plt.scatter(SVb[:, 0], SVb[:, 1],  s=80, facecolors='none', edgecolors='black')
# plt.scatter(SVu[:, 0], SVu[:, 1],  s=80, facecolors='none', edgecolors='red')
# # plt.savefig(name+' SVs ', format='png')
# plt.savefig(name+' decision ', format='png')

# # plt.close('all')
# plt.show()