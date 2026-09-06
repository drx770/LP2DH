function X = lp2dh_fibonacci_transform(X)
%LP2DH_FIBONACCI_TRANSFORM Historical signed Fibonacci-bin PDV map.

sign_value = sign(X);
absolute_value = abs(X);
mapped = zeros(size(X));
edges = [1 2 3 5 8 13 21 34 55 89 144 233];
mapped(absolute_value>0 & absolute_value<=1) = 1;
for k = 1:numel(edges)-1
    mapped(absolute_value>edges(k) & absolute_value<=edges(k+1)) = k+1;
end
mapped(absolute_value>edges(end)) = numel(edges)+1;
X = sign_value.*mapped;
end

