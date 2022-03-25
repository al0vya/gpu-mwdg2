function Z = diff_topo(X, Y)

cone1 = 1.0 - 0.2 * sqrt( ( X - 20.0 ) .* ( X - 20.0 ) + ( Y - 15.0 ) .* ( Y - 15.0 ) );
cone2 = 2.0 - 0.5 * sqrt( ( X - 40.0 ) .* ( X - 40.0 ) + ( Y - 15.0 ) .* ( Y - 15.0 ) );
cone3 = 3.0 - 0.3 * sqrt( ( X - 60.0 ) .* ( X - 60.0 ) + ( Y - 15.0 ) .* ( Y - 15.0 ) );

Z = 0;
Z = max(Z, cone1);
Z = max(Z, cone2);
Z = max(Z, cone3);

end