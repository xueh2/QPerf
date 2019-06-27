% example of QPerf perfusion flow mapping

clear all
close all

% suppose the QPerf repo was cloned to QPerf directory
% git clone ...

% load the stored data

cd .\QPerf\examples

load perf_data_stress

command = ['../software/QPerf/gadgetron_QPerf_mapping -f ./aif_stress -i ./data_stress -m ./MBF_stress --foot ' num2str(foot) ' --dt 500'];
dos(command);

% load and visualize map
fmap = analyze75read('MBF_stress');

figure; 
plot(aif);
title('AIF');

figure; imshow(fmap, 'DisplayRange', [0 6]);PerfColorMap;
