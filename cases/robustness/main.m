clc

eta_three_humps = 1;
eta_non_diff    = 1.78;
eta_diff        = 1.95;

plot_topo(@three_humps,   eta_three_humps, 'humps');
plot_topo(@non_diff_topo, eta_non_diff,    'non-diff');
plot_topo(@diff_topo,     eta_diff,        'diff');

exit()