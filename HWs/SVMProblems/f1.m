function z = f(x, y)
	x1 = [-1 -1];
	x2 = [-1 +1];
	x3 = [+1 -1];
	x4 = [+1 +1];
	z = zeros(100, 100);
	for i = 1:100
		for j = 1:100
			v = [x(i) y(j)];
			z(i, j) = sign((-K1(x1, v) + K1(x2, v) + K1(x3, v) - K1(x4, v))/4 - 1);
		end
	end
end