function [W,output] = lp2dh_optimize_W(W0,Q,linear)
%LP2DH_OPTIMIZE_W One Wen-Yin/Stiefel W update for the LP2DH objective.

options.record = 0;
options.mxitr = 100;
options.xtol = 1e-7;
options.gtol = 1e-6;
options.ftol = 1e-10;
options.tau = 1e-3;
[W,output] = OptStiefelGBB(W0,@objective,options,Q,linear);
end

function [value,gradient] = objective(W,Q,linear)
% Q contains the L4 locality, L1 quantization quadratic, L2 balance,
% and L3 variance terms. LINEAR is the current binary-code contribution.
value = trace(W'*Q*W)+sum(linear.*W,'all');
gradient = 2*Q*W+linear;
end
