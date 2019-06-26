% example of QPerf perfusion flow mapping

clear all
close all

% load the stored AIF signal

cd .\QPerf\examples

load perf_data

command = ['gadgetron_QPerf_mapping -f ./aif -i ./data -m ./MBF --foot ' num2str(foot) ' --peak ' num2str(peak) ' --dt 500'];
dos(command);

% load and visualize map
fmap = analyze75read('MBF');

figure; 
plot(aif);
title('AIF');

figure; imshow(fmap, 'DisplayRange', [0 8]);PerfColorMap;
