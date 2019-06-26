function perf_color_map = PerfColorMap(use_cmap)

if(nargin<1)
    use_cmap = 1;
end

load color_perf_flow_map2
perf_color_map = p;
if (use_cmap) 
    colormap(perf_color_map)
end