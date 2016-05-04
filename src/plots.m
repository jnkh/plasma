%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is part of the
% Disruption Predictor Feature Developer tools.
% These scripts were developed to utilize
% Support Vector Machines to evaluate diagnostic
% signals as features for predicting disruptions
%
% Developer: Matthew Parsons, mparsons@pppl.gov
%
% plots.m - This script is called from vars.m and
% plots the data output from test.m
% 
% 
% Inputs:
% out_dir/run.out
% out_dir/correct_alarms_binned.dat
% out_dir/false_alarms_d.dat
% out_dir/false_alarms_nd.dat
% out_dir/final_three.dat
% 
% Outputs:
% out_dir/correct_alarms_binned.png
% out_dir/false_alarms_d.png
% out_dir/false_alarms_nd.png
% out_dir/final_three.png
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






system('echo Making Plots');
%cd /u/mparsons/jet/test/DATA/2015-07-10/test_01

if(exist('out_dir'))
    cd(out_dir)
end

fid = fopen('run.out');
data = textscan(fid,'%s','Delimiter','\n');
p = findstr(data{1}{2},' ');
C = str2num(data{1}{2}(1:p));
gamma = str2num(data{1}{2}(p:end));
fclose(fid);


load correct_alarms_binned.dat
if(size(correct_alarms_binned,1) ~= 0)
    semilogx(-correct_alarms_binned(:,1),correct_alarms_binned(:,2),'LineWidth',4);
    hold on;
    plot([0.03,0.03],[0,1],'Color','green','LineWidth',3);
    title(sprintf('C = %f, gamma = %f',C,gamma),'FontSize',20);
    xlabel('(Time to Disruption) / (s)', 'FontSize',20);
    ylabel('Cumulative Fraction of Identified Disruptions', 'FontSize',20);
    set(gca,'FontSize',20);
    print('correct_alarms_binned','-dpng');
    correct_alarms_binned = [];
    hold off;
end


load false_alarms_d.dat;
if(size(false_alarms_d,1) ~= 0)
    plot(false_alarms_d(:,2)-false_alarms_d(:,3),false_alarms_d(:,1),'.');
    title(sprintf('C = %f, gamma = %f',C,gamma),'FontSize',20);
    xlabel('(Time - Disruption Time) / (s)', 'FontSize',20);
    ylabel('Shot', 'FontSize',20);
    set(gca,'FontSize',20);
    set(gca,'YTickLabel',num2str(get(gca,'YTick').'));
    print('false_alarms_d','-dpng');
    false_alarms_d = [];
end                               


load false_alarms_nd.dat;
if(size(false_alarms_nd,1) ~= 0)
    plot(false_alarms_nd(:,2),false_alarms_nd(:,1),'.');
    title(sprintf('C = %f, gamma = %f',C,gamma),'FontSize',20);
    xlabel('(Time) / (s)', 'FontSize',20);
    ylabel('Shot', 'FontSize',20);
    set(gca,'FontSize',20);
    set(gca,'YTickLabel',num2str(get(gca,'YTick').'));
    print('false_alarms_nd','-dpng');
    false_alarms_nd = [];
end                             


load final_three.dat
if(size(final_three,1) ~= 0)
    P1 = [final_three(:,3); final_three(:,8); final_three(:,13)];
    P2 = [final_three(:,4); final_three(:,9); final_three(:,14)];
    P3 = [final_three(:,5); final_three(:,10); final_three(:,15)];
    Cs = [final_three(:,1)*0+1; final_three(:,6)*0+2; final_three(:,11)*0+3];
    scatter3(P1,P2,P3,10,Cs,'*');
    title(sprintf('C = %f, gamma = %f',C,gamma),'FontSize',20);
    xlabel('Time / (s)', 'FontSize',20);
    ylabel('Shot', 'FontSize',20);
    set(gca,'FontSize',20);
    set(gca,'YTickLabel',num2str(get(gca,'YTick').'));
    print('final_three','-dpng');
end  
