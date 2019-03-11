function z = K2(x, y)
	z = exp(-sum((x - y).^2));
end