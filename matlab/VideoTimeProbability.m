clear;
close all;
clc;
load 'VideoWorkspace.mat';
plot(videoinferences,frames,'r', 0.5, false)

function plot(table,frames,color, optimal, thresholds)
    figure();
    table_timestamps = table2array(table(:,1));
    table_timestamps = table_timestamps.';
    table_scores = table2array(table(:,2));
    table_scores = table_scores.';
    %stairs(table_timestamps,table_scores, color,'LineWidth',2)
    area(table_timestamps,table_scores)
    labels = table2array(frames(:,2));
    labels = labels.';
    xlabel("Time")
    ylabel("Fire Probability")
    xticks(labels)
    ylim([0 1])
    labelnames = table2array(frames(2:end,3));
    labelnames = labelnames.';
    xticklabels(labelnames);
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

