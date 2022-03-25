function Z = three_humps(X, Y)

x_1 = 30;
y_1 = 6;
x_2 = 30;
y_2 = 24;
x_3 = 47.5;
y_3 = 15;

rm_1 = 8;
rm_2 = 8;
rm_3 = 10;

r_1 = sqrt( (X - x_1) .* (X - x_1) + (Y - y_1) .* (Y - y_1) );
r_2 = sqrt( (X - x_2) .* (X - x_2) + (Y - y_2) .* (Y - y_2) );
r_3 = sqrt( (X - x_3) .* (X - x_3) + (Y - y_3) .* (Y - y_3) );

zb_1 = (rm_1 - r_1) / 8;
zb_2 = (rm_2 - r_2) / 8;
zb_3 = 0.3 * (rm_3 - r_3);

Z = max(zb_1, zb_2);
Z = max(Z, zb_3);
Z = max(Z, 0);

end