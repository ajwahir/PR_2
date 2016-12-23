function [reduced]=principalcomp(data,l)

mu_data=mean(data);
ybar=[];

for ii=1:size(data,1)
    ybar=[ybar;data(ii,:)-mu_data];
end



C=(ybar'*ybar)./(size(ybar,1));
e=eig(C);
J=sum(e(1:end-l))
[V D]=eig(C);
Jdash=sum(e)
var=(1-J/Jdash)*100
Q= V(:,end-l+1:end);
reduced=data*Q;

end



