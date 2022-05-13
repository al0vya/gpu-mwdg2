#include "get_limited_slopes.cuh"

__device__
Slopes get_limited_slopes
(
	const PlanarCoefficients& u,
	const PlanarCoefficients& u_n,
	const PlanarCoefficients& u_e,
	const PlanarCoefficients& u_s,
	const PlanarCoefficients& u_w,
	const LegendreBasis&      basis_n,
	const LegendreBasis&      basis_e,
	const LegendreBasis&      basis_s,
	const LegendreBasis&      basis_w,
	const real&               dx_finest,
	const real&               dy_finest,
	const real&               tol_Krivo
)
{
	// ------- //
	// STEP 3b //
	// ------- //
	const real u_n_pos = eval_loc_face_val_dg2(u_n, basis_n);
	const real u_e_pos = eval_loc_face_val_dg2(u_e, basis_e);
	const real u_s_neg = eval_loc_face_val_dg2(u_s, basis_s);
	const real u_w_neg = eval_loc_face_val_dg2(u_w, basis_w);

	const real u_n_neg = eval_loc_face_val_dg2(u, { C(1.0), C(0.0), sqrt( C(3.0) ) } );
	const real u_e_neg = eval_loc_face_val_dg2(u, { C(1.0), sqrt( C(3.0) ), C(0.0) } );
	const real u_s_pos = eval_loc_face_val_dg2(u, { C(1.0), C(0.0), -sqrt( C(3.0) ) });
	const real u_w_pos = eval_loc_face_val_dg2(u, { C(1.0), -sqrt( C(3.0) ), C(0.0) });

	// ------- //
	// STEP 3c //
	// ------- //
	const real jump_n = abs(u_n_pos - u_n_neg);
	const real jump_e = abs(u_e_pos - u_e_neg);
	const real jump_s = abs(u_s_pos - u_s_neg);
	const real jump_w = abs(u_w_pos - u_w_neg);

	// ------ //
	// STEP 4 //
	// ------ //
	const real norm_x = max( abs(u._0 - u_w._0), abs(u_e._0 - u._0) );
	const real norm_y = max( abs(u._0 - u_s._0), abs(u_n._0 - u._0) );

	// ------ //
	// STEP 5 //
	// ------ //
	const real DS_e = ( norm_x > C(1e-12) ) ? jump_e / (C(0.5) * dx_finest * norm_x) : C(0.0);
	const real DS_w = ( norm_x > C(1e-12) ) ? jump_w / (C(0.5) * dx_finest * norm_x) : C(0.0);
					    				  
	const real DS_n = ( norm_y > C(1e-12) ) ? jump_n / (C(0.5) * dy_finest * norm_y) : C(0.0);
	const real DS_s = ( norm_y > C(1e-12) ) ? jump_s / (C(0.5) * dy_finest * norm_y) : C(0.0);

	const real u1x_limited = (max(DS_e, DS_w) < tol_Krivo) ? u._1x : minmod
	(
		u._1x,
		(u._0 - u_w._0) / sqrt( C(3.0) ),
		(u_e._0 - u._0) / sqrt( C(3.0) )
	);

	const real u1y_limited = (max(DS_n, DS_s) < tol_Krivo) ? u._1y : minmod
	(
		u._1y,
		(u._0 - u_s._0) / sqrt( C(3.0) ),
		(u_n._0 - u._0) / sqrt( C(3.0) )
	);

	return { u1x_limited, u1y_limited };
}