
load('Use_Audio512.mat') ;  %��ȡ����
a = 1:1:126;    a = a';    set = sprintfc('%g',a);   
W = size(set,1);
 
for i=2:W
data =eval(['Use_Audio' num2str(i)]);
data = data';

variable = strcat('inpot_Audio',set{i});   
eval([ variable , '=','data',';']);
%save('intput.mat',variable);
save('intput.mat',variable,'-append');
%--$$$$$$$$$$$$$$$$$$$    ---������---   $$$$$$$$$$$$$$$$$$$ --
m=length(data(1,:));
n=0.5+0.25*randn(length(data(:,1)),m);
y_out = n+data;

 %���������磺Use_Audio002
 variable = strcat('OUT_Audio',set{i});   
eval([ variable , '=','y_out',';']);
%save('output.mat',variable);
save('output.mat',variable,'-append');

end
