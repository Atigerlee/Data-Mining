clear;
clc;

h1 = figure('Position',[90 50 450 500],'MenuBar','none','Name','Figure');
in = fopen("plot.txt",'r');
data = fscanf(in,'%lf');
fclose(in);
elements = length(data);
x = 2:(elements + 1);
y = data;
% 樣式、填充顏色、線條粗細
plot(x,y,'-ow','MarkerFaceColor','w','LineWidth',1.5);
title('WCSS');
xlabel('Number of Clusters(K)');
ylabel('WCSS Value');
grid on;