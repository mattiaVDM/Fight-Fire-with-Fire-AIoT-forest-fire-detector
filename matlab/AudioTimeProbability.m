clear;
close all;
clc;
load 'AudioWorkspace.mat';

plotmodel(MFECONV1D2Cmat,frames,'r', 0.71, true) %0.469
%plotmodel(MFECONV2D2Cmat,frames,'c', 0.387, true)
%plotmodel(SPECTROCONV1D2Cmat,frames, 'g', 0.102 , true)

%plotmodel(MFECONV1D4Cmat,frames,'b', 0.031, true)
%plotmodel(SPECTROCONV2D4CV1mat,frames, 'm',0.059, true)
%plotmodel(SPECTROCONV2D4CV2mat,frames, 'k',0.28, true) %0.387

function plotmodel(table,frames,color, optimal, thresholds)
    figure();
    table_timestamps = table2array(table(:,2));
    table_timestamps = table_timestamps.';
    table_scores = table2array(table(:,3));
    table_scores = table_scores.';
    %stairs(table_timestamps,table_scores, color,'LineWidth',2)
    area(table_timestamps,table_scores)
    labels = table2array(frames(2:end,2));
    labels = labels.';
    xlabel("Time")
    ylabel("Fire Probability")
    xticks(labels)
    ylim([0 1])
    labelnames = table2array(frames(2:end,3));
    labelnames = labelnames.';
    xticklabels(labelnames);
    title(table.VarName4(1))
    ax = gca;
    ax.XGrid = 'on';    
    hold on
    if thresholds == true
        yline(0.5,'r')
        hold on
        yline(optimal,'b')
    end
    hold off
end

