clear;
fileID = fopen('data/lin_sep_1/class1_train.txt','r');
formatSpec = '%f';
d1_train = fscanf(fileID,formatSpec);
d1_train=reshape(d1_train,2,length(d1_train)/2)';

fileID = fopen('data/lin_sep_1/class1_val.txt','r');
formatSpec = '%f';
d1_val = fscanf(fileID,formatSpec);
d1_val=reshape(d1_val,2,length(d1_val)/2)';

fileID = fopen('data/lin_sep_1/class1_test.txt','r');
formatSpec = '%f';
d1_test = fscanf(fileID,formatSpec);
d1_test=reshape(d1_test,2,length(d1_test)/2)';

fileID = fopen('data/lin_sep_1/class2_train.txt','r');
formatSpec = '%f';
d2_train = fscanf(fileID,formatSpec);
d2_train=reshape(d2_train,2,length(d2_train)/2)';

fileID = fopen('data/lin_sep_1/class2_val.txt','r');
formatSpec = '%f';
d2_val = fscanf(fileID,formatSpec);
d2_val=reshape(d2_val,2,length(d2_val)/2)';

fileID = fopen('data/lin_sep_1/class2_test.txt','r');
formatSpec = '%f';
d2_test = fscanf(fileID,formatSpec);
d2_test=reshape(d2_test,2,length(d2_test)/2)';

fileID = fopen('data/lin_sep_1/class3_train.txt','r');
formatSpec = '%f';
d3_train = fscanf(fileID,formatSpec);
d3_train=reshape(d3_train,2,length(d3_train)/2)';

fileID = fopen('data/lin_sep_1/class3_val.txt','r');
formatSpec = '%f';
d3_val = fscanf(fileID,formatSpec);
d3_val=reshape(d3_val,2,length(d3_val)/2)';

fileID = fopen('data/lin_sep_1/class3_test.txt','r');
formatSpec = '%f';
d3_test = fscanf(fileID,formatSpec);
d3_test=reshape(d3_test,2,length(d3_test)/2)';


fileID = fopen('data/lin_sep_1/class4_train.txt','r');
formatSpec = '%f';
d4_train = fscanf(fileID,formatSpec);
d4_train=reshape(d4_train,2,length(d4_train)/2)';

fileID = fopen('data/lin_sep_1/class4_val.txt','r');
formatSpec = '%f';
d4_val = fscanf(fileID,formatSpec);
d4_val=reshape(d4_val,2,length(d4_val)/2)';

fileID = fopen('data/lin_sep_1/class4_test.txt','r');
formatSpec = '%f';
d4_test = fscanf(fileID,formatSpec);
d4_test=reshape(d4_test,2,length(d4_test)/2)';

train_data = vertcat(d1_train,d2_train,d3_train,d4_train);
val_data = vertcat(d1_val,d2_val,d3_val,d4_val);
test_data = vertcat(d1_test,d2_test,d3_test,d4_test);

n=size(train_data,1);
    
act_class=ones(n,1);
for i=1:4
    for j=1:n/4
        act_class((i-1)*n/4+j,1)=i;
    end
end

% comparing different classes taken 2 at a time

train_data1 = vertcat(d1_train,d2_train);
train_data2 = vertcat(d1_train,d3_train);
train_data3 = vertcat(d1_train,d4_train);
train_data4 = vertcat(d2_train,d3_train);
train_data5 = vertcat(d2_train,d4_train);
train_data6 = vertcat(d3_train,d4_train);

c1=fitcsvm(train_data1,[act_class(1:250);act_class(251:500)],'KernelFunction','linear','Standardize',true,'ClassNames',[1,2]);
c2=fitcsvm(train_data2,[act_class(1:250);act_class(501:750)],'KernelFunction','linear','Standardize',true,'ClassNames',[1,3]);
c3=fitcsvm(train_data3,[act_class(1:250);act_class(751:1000)],'KernelFunction','linear','Standardize',true,'ClassNames',[1,4]);
c4=fitcsvm(train_data4,[act_class(251:500);act_class(501:750)],'KernelFunction','linear','Standardize',true,'ClassNames',[2,3]);
c5=fitcsvm(train_data5,[act_class(251:500);act_class(751:1000)],'KernelFunction','linear','Standardize',true,'ClassNames',[2,4]);
c6=fitcsvm(train_data6,[act_class(501:750);act_class(751:1000)],'KernelFunction','linear','Standardize',true,'ClassNames',[3,4]);


[~,scores1]=predict(c1,val_data);
[~,scores2]=predict(c2,val_data);
[~,scores3]=predict(c3,val_data);
[~,scores4]=predict(c4,val_data);
[~,scores5]=predict(c5,val_data);
[~,scores6]=predict(c6,val_data);

pred_class1=zeros(600,1);
pred_class2=zeros(600,1);
pred_class3=zeros(600,1);
pred_class4=zeros(600,1);

for i=1:600
   pred_class1(i,1)=scores1(i,1)+scores2(i,1)+scores3(i,1);
   pred_class2(i,1)=scores1(i,2)+scores4(i,1)+scores5(i,1);
   pred_class3(i,1)=scores2(i,2)+scores4(i,2)+scores6(i,1);
   pred_class4(i,1)=scores3(i,2)+scores5(i,2)+scores6(i,2);
end


n=size(val_data,1);
act_class=ones(n,1);
for i=1:4
    for j=1:n/4
        act_class((i-1)*n/4+j,1)=i;
    end
end

pred_class=ones(n,1);

for i=1:n
        if pred_class1(i,1)>=pred_class2(i,1)& pred_class1(i,1)>=pred_class3(i,1)& pred_class1(i,1)>=pred_class4(i,1)
            pred_class(i,1)=1;
        elseif pred_class2(i,1)>=pred_class3(i,1)& pred_class2(i,1)>=pred_class4(i,1)
            pred_class(i,1)=2;
        elseif pred_class3(i,1)>=pred_class4(i,1)
            pred_class(i,1)=3;
        else
            pred_class(i,1)=4;
        end
end
confusionmat(a,predicted_values(1:n,k_opt))

b=predicted_values(1:n,k_opt);

t=zeros(4,length(a));
y=zeros(4,length(a));
t(1,:)=(a==1);
t(2,:)=(a==2);
t(3,:)=(a==3);
t(4,:)=(a==4);
y(1,:)=(b==1);
y(2,:)=(b==2);
y(3,:)=(b==3);
y(4,:)=(b==4);
plotconfusion(t,y)

%plotting decision surface

min_train=min(train_data);
max_train=max(train_data);
xrange=[min_train(1)-1 max_train(1)+1];
yrange=[min_train(2)-1 max_train(2)+1];
inc = 0.5;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];

%finding class label at each point of gridmatrix
n=size(train_data,1);
    
a=ones(n,2);
for i=1:n
    a(i,1)=0;
end

for i=1:4
    for j=1:n/4
        a((i-1)*n/4+j,2)=i;
    end
end

grid_data_neighbours = zeros(size(xy,1),5);
for i=1:size(xy,1)
    for j=1:size(train_data,1)
        a(j,1)=sqrt((xy(i,1)-train_data(j,1))^2+(xy(i,2)-train_data(j,2))^2);
    end
    train_data_frame=sortrows(a,1);
    grid_data_neighbours(i,1:5)=train_data_frame(1:5,2);
end
        
idx=zeros(size(xy,1),1);


for i=1:size(xy,1)
    idx(i,1)=mode(grid_data_neighbours(i,1:5));
end

decisionmap = reshape(idx, image_size);
figure;
 
%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue, 4=white 
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1,; 1 1 1]
colormap(cmap);


% plot the class training data.
plot(d1_train(:,1),d1_train(:,2), 'r.');
plot(d2_train(:,1),d2_train(:,2), 'go');
plot(d3_train(:,1),d3_train(:,2), 'b*');
plot(d4_train(:,1),d4_train(:,2), '');

% include legend
legend('Class 1', 'Class 2', 'Class 3', 'Class 4','Location','NorthOutside', ...
    'Orientation', 'horizontal');
 
% label the axes.
xlabel('x');
ylabel('y');




