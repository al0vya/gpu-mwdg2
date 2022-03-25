function Z = non_diff_topo(X, Y)

Y_lower_bound = 11 <= Y;
Y_upper_bound = 19 >= Y;

Y_bound = Y_lower_bound .* Y_upper_bound;

X_lower_bound_1 = 16 <= X;
X_upper_bound_1 = 24 >= X;

X_lower_bound_2 = 36 <= X;
X_upper_bound_2 = 44 >= X;

X_lower_bound_3 = 56 <= X;
X_upper_bound_3 = 64 >= X;

X_1 = X_lower_bound_1 .* X_upper_bound_1;
X_2 = X_lower_bound_2 .* X_upper_bound_2;
X_3 = X_lower_bound_3 .* X_upper_bound_3;

block_1 = X_1 .* Y_bound * 0.86;
block_2 = X_2 .* Y_bound * 1.78;
block_3 = X_3 .* Y_bound * 2.30;

Z = block_1 + block_2 + block_3;

end