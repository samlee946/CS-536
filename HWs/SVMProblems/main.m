x = linspace(-1,1);
y = linspace(-1,1);
z1 = f1(x, y);
z2 = f2(x, y);
figure
surf(x, y, z1);
colormap
figure
surf(x, y, z2);
colormap
