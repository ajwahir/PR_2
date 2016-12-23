
load('G:\5th_Semester\PatternRecognition-CS6690\assignment1\Dataset_Assignment1-20161026T193723Z\Dataset_Assignment1\Dataset-2_real_world\a_Image Classification data\CompleteData.mat')
prev=1;
class1full=fullpca_20(1:lengths(1),:);
prev=prev+lengths(1);
class2full=fullpca_20(prev:prev+lengths(2)-1,:);
prev=prev+lengths(2);
class3full=fullpca_20(prev:prev+lengths(3)-1,:);
prev=prev+lengths(3);
class4full=fullpca_20(prev:prev+lengths(4)-1,:);
prev=prev+lengths(4);
class5full=fullpca_20(prev:prev+lengths(5)-1,:);
% prev=prev+size(class1full,5)


% Reading all the classes 
% class1full=CompleteData{10,1};
% class2full=CompleteData{15,1};
% class3full=CompleteData{17,1};
% class4full=CompleteData{7,1};
% class5full=CompleteData{13,1};

fulldata=[class1full;class2full;class3full;class4full;class5full];
% fullpca=principalcomp(fulldata,46);
% lengths=[size(class1full,1) size(class2full,1) size(class3full,1) size(class4full,1) size(class5full,1)];
% 
% save('fullpca.mat','fullpca');
% save('fulldata.mat','fulldata');
% save('lengths.mat','lengths');

% extracting train test and val from class1
class1train=class1full(1:floor(size(class1full,1)*0.70),:);
class1val=class1full(floor(size(class1full,1)*0.70)+1:floor(size(class1full,1)*.15)+floor(size(class1full,1)*0.70),:);
class1test=class1full(floor(size(class1full,1)*.15)+floor(size(class1full,1)*0.70)+1:end,:);

% extracting train test and val from class2
class2train=class2full(1:floor(size(class2full,1)*0.70),:);
class2val=class2full(floor(size(class2full,1)*0.70)+1:floor(size(class2full,1)*.15)+floor(size(class2full,1)*0.70),:);
class2test=class2full(floor(size(class2full,1)*.15)+floor(size(class2full,1)*0.70)+1:end,:);

% extracting train test and val from class3
class3train=class3full(1:floor(size(class3full,1)*0.70),:);
class3val=class3full(floor(size(class3full,1)*0.70)+1:floor(size(class3full,1)*.15)+floor(size(class3full,1)*0.70),:);
class3test=class3full(floor(size(class3full,1)*.15)+floor(size(class3full,1)*0.70)+1:end,:);

% extracting train test and val from class4
class4train=class4full(1:floor(size(class4full,1)*0.70),:);
class4val=class4full(floor(size(class4full,1)*0.70)+1:floor(size(class4full,1)*.15)+floor(size(class4full,1)*0.70),:);
class4test=class4full(floor(size(class4full,1)*.15)+floor(size(class4full,1)*0.70)+1:end,:);

% extracting train test and val from class5
class5train=class5full(1:floor(size(class5full,1)*0.70),:);
class5val=class5full(floor(size(class5full,1)*0.70)+1:floor(size(class5full,1)*.15)+floor(size(class5full,1)*0.70),:);
class5test=class5full(floor(size(class5full,1)*.15)+floor(size(class5full,1)*0.70)+1:end,:);

fulltrain=[class1train;class2train;class3train;class4train;class5train];
trainlengths=[size(class1train,1);size(class2train,1);size(class3train,1);size(class4train,1);size(class5train,1)];

fulltest=[class1train;class2train;class3train;class4train;class5train];
trainlengths=[size(class1train,1);size(class2train,1);size(class3train,1);size(class4train,1);size(class5train,1)];

fulltrain=[class1train;class2train;class3train;class4train;class5train];
trainlengths=[size(class1train,1);size(class2train,1);size(class3train,1);size(class4train,1);size(class5train,1)];

fulltrainforg=[horzcat(class1train,ones(size(class1train,1),1));horzcat(class2train,2*ones(size(class2train,1),1));horzcat(class3train,3*ones(size(class3train,1),1));horzcat(class4train,4*ones(size(class4train,1),1));horzcat(class5train,5*ones(size(class5train,1),1))];
fulltestforg=[horzcat(class1test,ones(size(class1test,1),1));horzcat(class2test,2*ones(size(class2test,1),1));horzcat(class3test,3*ones(size(class3test,1),1));horzcat(class4test,4*ones(size(class4test,1),1));horzcat(class5test,5*ones(size(class5test,1),1))];
fullvalforg=[horzcat(class1val,ones(size(class1val,1),1));horzcat(class2val,2*ones(size(class2val,1),1));horzcat(class3val,3*ones(size(class3val,1),1));horzcat(class4val,4*ones(size(class4val,1),1));horzcat(class5val,5*ones(size(class5val,1),1))];


%% 

fulldata=[class1full;class2full;class3full;class4full;class5full];
fullpca_31=principalcomp(fulldata,31);
fullpca_25=principalcomp(fulldata,25);
fullpca_20=principalcomp(fulldata,20);

save('fullpca_31.mat','fullpca_31')
save('fullpca_25.mat','fullpca_25')
save('fullpca_20.mat','fullpca_20')

