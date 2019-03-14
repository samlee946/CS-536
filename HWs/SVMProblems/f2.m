function z = f(x, y)
	x1 = [-1 -1];
	x2 = [-1 +1];
	x3 = [+1 -1];
	x4 = [+1 +1];
	z = zeros(100, 100);
	for i = 1:100
		for j = 1:100
			v = [x(i) y(j)];
			z(i, j) = sign(exp(8)/(exp(4) - 1).^2 * (-K2(x1, v) + K2(x2, v) + K2(x3, v) - K2(x4, v)) - 0.075326);
		end
	end
end