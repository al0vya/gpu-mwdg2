#include "add_ghost_cells.cuh"

__global__
void add_ghost_cells
(
	AssembledSolution d_assem_sol,
	Neighbours        d_neighbours,
	SolverParams      solver_params,
	SimulationParams  sim_params,
	Boundaries        boundaries,
	real              time_now,
	real              dt,
	real              dx_finest,
	int               test_case
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= d_assem_sol.length) return;

	int level = d_assem_sol.levels[idx];

	MortonCode code = d_assem_sol.act_idcs[idx] - get_lvl_idx(level);

	Coordinate x = compact(code);
	Coordinate y = compact(code >> 1);
	
	bool finest  = (level == solver_params.L);
	bool bound   = false;
	bool border  = false;
	bool finebdy = false;

	bool mwdg2 = (solver_params.solver_type == MWDG2);

	bool flow_EW = false;
	bool flow_NS = false;
	
	switch (test_case)
	{
	    case 1:    // wet 1D c property
	    case 3:    // wet/dry 1D c property
	    case 5:    // wet dam break
	    case 7:    // dry dam break
	    case 9:    // dry dam break with friction
	    case 11:   // wet building overtopping
	    case 13:   // dry building overtopping
	    case 15:   // triangular dam break
	    case 17:   // parabolic bowl
	    	flow_EW = true;
	    	break;
	    case 2: 
	    case 4: 
	    case 6: 
	    case 8: 
	    case 10:
	    case 12:
	    case 14:
	    case 16:
	    case 18:
	    	flow_NS = true;
	    	break;
	    default:
	    	break;
	}

	if (d_neighbours.north.act_idcs[idx] == -1)
	{
		d_neighbours.north.h0[idx]  = d_assem_sol.h0[idx];
		d_neighbours.north.qx0[idx] = d_assem_sol.qx0[idx];
		d_neighbours.north.qy0[idx] = (test_case == 0 || test_case == 20) ? C(0.0) : d_assem_sol.qy0[idx];
		d_neighbours.north.z0[idx]  = d_assem_sol.z0[idx];
		
		if (mwdg2)
		{
			if (test_case == 0)
			{
				d_neighbours.north.h1x[idx]  = C(0.0);
				d_neighbours.north.qx1x[idx] = C(0.0);
				d_neighbours.north.qy1x[idx] = C(0.0);
				d_neighbours.north.z1x[idx]  = C(0.0);
				
				d_neighbours.north.h1y[idx]  = C(0.0);
				d_neighbours.north.qx1y[idx] = C(0.0);
				d_neighbours.north.qy1y[idx] = C(0.0);
				d_neighbours.north.z1y[idx]  = C(0.0);
			}
			else
			{
				if ( flow_NS || (!flow_NS && !flow_EW) )
				{
					if (!flow_NS && !flow_EW)
					{
						d_assem_sol.h1y[idx]  = C(0.0);
						d_assem_sol.qx1y[idx] = C(0.0);
						d_assem_sol.qy1y[idx] = C(0.0);
						d_assem_sol.z1y[idx]  = C(0.0);
					}

					d_assem_sol.h1x[idx]  = C(0.0);
					d_assem_sol.qx1x[idx] = C(0.0);
					d_assem_sol.qy1x[idx] = C(0.0);
					d_assem_sol.z1x[idx]  = C(0.0);

					d_neighbours.north.h1x[idx]  = C(0.0);
					d_neighbours.north.qx1x[idx] = C(0.0);
					d_neighbours.north.qy1x[idx] = C(0.0);
					d_neighbours.north.z1x[idx]  = C(0.0);

					d_neighbours.north.h1y[idx]  = C(0.0);
					d_neighbours.north.qx1y[idx] = C(0.0);
					d_neighbours.north.qy1y[idx] = C(0.0);
					d_neighbours.north.z1y[idx]  = C(0.0);
				}

				if (flow_EW)
				{
					d_neighbours.north.h1x[idx]  = d_assem_sol.h1x[idx];
					d_neighbours.north.qx1x[idx] = d_assem_sol.qx1x[idx];
					d_neighbours.north.qy1x[idx] = d_assem_sol.qy1x[idx];
					d_neighbours.north.z1x[idx]  = d_assem_sol.z1x[idx];

					d_neighbours.north.h1y[idx]  = C(0.0);
					d_neighbours.north.qx1y[idx] = C(0.0);
					d_neighbours.north.qy1y[idx] = C(0.0);
					d_neighbours.north.z1y[idx]  = C(0.0);
				}
			}	
		}

		bound  = ( boundaries.north.bound(x) );
		border = ( y == sim_params.ysz - 1 );

		finebdy = (finest && bound && border);

		if (finebdy && test_case == 0)
		{
			if (boundaries.north.bdytype == FREE)
			{
				d_neighbours.north.h0[idx]  = pow( (d_assem_sol.qy0[idx] * d_assem_sol.qy0[idx]) / sim_params.g, C(1.0) / C(3.0) );
				d_neighbours.north.qx0[idx] = d_assem_sol.qx0[idx];
				d_neighbours.north.qy0[idx] = d_assem_sol.qy0[idx];
			}
			else if
			(
				boundaries.north.bdytype == HFIX
				|| 
				boundaries.north.bdytype == HVAR
			)
			{
				d_assem_sol.h0[idx]        = boundaries.north.inlet - d_assem_sol.z0[idx];
				d_neighbours.north.h0[idx] = d_assem_sol.h0[idx];
			}
			else if
			(
				boundaries.north.bdytype == QFIX
				|| 
				boundaries.north.bdytype == QVAR
			)
			{
				d_assem_sol.h0[idx]        = boundaries.north.q_src(dt, dx_finest);
				d_neighbours.north.h0[idx] = d_assem_sol.h0[idx];
			}
		}
	}

	if (d_neighbours.east.act_idcs[idx] == -1)
	{
		d_neighbours.east.h0[idx]  = d_assem_sol.h0[idx];
		d_neighbours.east.qx0[idx] = (test_case == 0 || test_case == 20) ? C(0.0) : d_assem_sol.qx0[idx];
		d_neighbours.east.qy0[idx] = d_assem_sol.qy0[idx];
		d_neighbours.east.z0[idx]  = d_assem_sol.z0[idx];
		
		if (mwdg2)
		{
			if (test_case == 0)
			{
				d_neighbours.east.h1x[idx]  = C(0.0);
				d_neighbours.east.qx1x[idx] = C(0.0);
				d_neighbours.east.qy1x[idx] = C(0.0);
				d_neighbours.east.z1x[idx]  = C(0.0);
				
				d_neighbours.east.h1y[idx]  = C(0.0);
				d_neighbours.east.qx1y[idx] = C(0.0);
				d_neighbours.east.qy1y[idx] = C(0.0);
				d_neighbours.east.z1y[idx]  = C(0.0);
			}
			else
			{
				if (flow_NS)
				{
					d_neighbours.east.h1y[idx]  = d_assem_sol.h1y[idx];
					d_neighbours.east.qx1y[idx] = d_assem_sol.qx1y[idx];
					d_neighbours.east.qy1y[idx] = d_assem_sol.qy1y[idx];
					d_neighbours.east.z1y[idx]  = d_assem_sol.z1y[idx];

					d_neighbours.east.h1x[idx]  = C(0.0);
					d_neighbours.east.qx1x[idx] = C(0.0);
					d_neighbours.east.qy1x[idx] = C(0.0);
					d_neighbours.east.z1x[idx]  = C(0.0);
				}

				if ( flow_EW || (!flow_NS && !flow_EW) )
				{
					if (!flow_NS && !flow_EW)
					{
						d_assem_sol.h1x[idx]  = C(0.0);
						d_assem_sol.qx1x[idx] = C(0.0);
						d_assem_sol.qy1x[idx] = C(0.0);
						d_assem_sol.z1x[idx]  = C(0.0);
					}
					
					d_assem_sol.h1y[idx]  = C(0.0);
					d_assem_sol.qx1y[idx] = C(0.0);
					d_assem_sol.qy1y[idx] = C(0.0);
					d_assem_sol.z1y[idx]  = C(0.0);

					d_neighbours.east.h1x[idx]  = C(0.0);
					d_neighbours.east.qx1x[idx] = C(0.0);
					d_neighbours.east.qy1x[idx] = C(0.0);
					d_neighbours.east.z1x[idx]  = C(0.0);

					d_neighbours.east.h1y[idx]  = C(0.0);
					d_neighbours.east.qx1y[idx] = C(0.0);
					d_neighbours.east.qy1y[idx] = C(0.0);
					d_neighbours.east.z1y[idx]  = C(0.0);
				}
			}	
		}

		bound  = ( boundaries.east.bound(y) );
		border = ( x == sim_params.xsz - 1 );

		finebdy = (finest && bound && border);

		if (finebdy && test_case == 0)
		{
			if (boundaries.east.bdytype == FREE)
			{
				d_neighbours.east.h0[idx]  = pow( (d_assem_sol.qx0[idx] * d_assem_sol.qx0[idx]) / sim_params.g, C(1.0) / C(3.0) );
				d_neighbours.east.qx0[idx] = d_assem_sol.qx0[idx];
				d_neighbours.east.qy0[idx] = d_assem_sol.qy0[idx];
			}
			else if
			(
				boundaries.east.bdytype == HFIX
				|| 
				boundaries.east.bdytype == HVAR
			)
			{
				d_assem_sol.h0[idx]       = boundaries.east.inlet - d_assem_sol.z0[idx];
				d_neighbours.east.h0[idx] = d_assem_sol.h0[idx];
			}
			else if
			(
				boundaries.east.bdytype == QFIX
				|| 
				boundaries.east.bdytype == QVAR
			)
			{
				d_assem_sol.h0[idx]       = boundaries.east.q_src(dt, dx_finest);
				d_neighbours.east.h0[idx] = d_assem_sol.h0[idx];
			}
		}
	}

	if (d_neighbours.south.act_idcs[idx] == -1)
	{
		d_neighbours.south.h0[idx]  = d_assem_sol.h0[idx];
		d_neighbours.south.qx0[idx] = d_assem_sol.qx0[idx];
		d_neighbours.south.qy0[idx] = (test_case == 0 || test_case == 20 || test_case == 16) ? C(0.0) : d_assem_sol.qy0[idx];
		d_neighbours.south.z0[idx]  = d_assem_sol.z0[idx];
		
		if (mwdg2)
		{
			if (test_case == 0)
			{
				d_neighbours.south.h1x[idx]  = C(0.0);
				d_neighbours.south.qx1x[idx] = C(0.0);
				d_neighbours.south.qy1x[idx] = C(0.0);
				d_neighbours.south.z1x[idx]  = C(0.0);
				
				d_neighbours.south.h1y[idx]  = C(0.0);
				d_neighbours.south.qx1y[idx] = C(0.0);
				d_neighbours.south.qy1y[idx] = C(0.0);
				d_neighbours.south.z1y[idx]  = C(0.0);
			}
			else
			{
				if ( flow_NS || (!flow_NS && !flow_EW) )
				{
					if (!flow_NS && !flow_EW)
					{
						d_assem_sol.h1y[idx]  = C(0.0);
						d_assem_sol.qx1y[idx] = C(0.0);
						d_assem_sol.qy1y[idx] = C(0.0);
						d_assem_sol.z1y[idx]  = C(0.0);
					}
					
					d_assem_sol.h1x[idx]  = C(0.0);
					d_assem_sol.qx1x[idx] = C(0.0);
					d_assem_sol.qy1x[idx] = C(0.0);
					d_assem_sol.z1x[idx]  = C(0.0);

					d_neighbours.south.h1x[idx]  = C(0.0);
					d_neighbours.south.qx1x[idx] = C(0.0);
					d_neighbours.south.qy1x[idx] = C(0.0);
					d_neighbours.south.z1x[idx]  = C(0.0);

					d_neighbours.south.h1y[idx]  = C(0.0);
					d_neighbours.south.qx1y[idx] = C(0.0);
					d_neighbours.south.qy1y[idx] = C(0.0);
					d_neighbours.south.z1y[idx]  = C(0.0);
				}

				if (flow_EW)
				{
					d_neighbours.south.h1x[idx]  = d_assem_sol.h1x[idx];
					d_neighbours.south.qx1x[idx] = d_assem_sol.qx1x[idx];
					d_neighbours.south.qy1x[idx] = d_assem_sol.qy1x[idx];
					d_neighbours.south.z1x[idx]  = d_assem_sol.z1x[idx];
					
					d_neighbours.south.h1y[idx]  = C(0.0);
					d_neighbours.south.qx1y[idx] = C(0.0);
					d_neighbours.south.qy1y[idx] = C(0.0);
					d_neighbours.south.z1y[idx]  = C(0.0);
				}
			}	
		}

		bound  = ( boundaries.south.bound(x) );
		border = ( y == 0 );

		finebdy = (finest && bound && border);

		if (finebdy && test_case == 0)
		{
			if (boundaries.south.bdytype == FREE)
			{
				d_neighbours.south.h0[idx]  = pow( (d_assem_sol.qy0[idx] * d_assem_sol.qy0[idx]) / sim_params.g, C(1.0) / C(3.0) );
				d_neighbours.south.qx0[idx] = d_assem_sol.qx0[idx];
				d_neighbours.south.qy0[idx] = d_assem_sol.qy0[idx];
			}
			else if
			(
				boundaries.south.bdytype == HFIX
				|| 
				boundaries.south.bdytype == HVAR
			)
			{
				d_assem_sol.h0[idx]        = boundaries.south.inlet - d_assem_sol.z0[idx];
				d_neighbours.south.h0[idx] = d_assem_sol.h0[idx];
			}
			else if
			(
				boundaries.south.bdytype == QFIX
				|| 
				boundaries.south.bdytype == QVAR
			)
			{
				d_assem_sol.h0[idx]        = boundaries.south.q_src(dt, dx_finest);
				d_neighbours.south.h0[idx] = d_assem_sol.h0[idx];
			}
		}
	}

	if (d_neighbours.west.act_idcs[idx] == -1)
	{
		d_neighbours.west.h0[idx]  = d_assem_sol.h0[idx];
		d_neighbours.west.qx0[idx] = (test_case == 0 || test_case == 20 || test_case == 15) ? C(0.0) : d_assem_sol.qx0[idx];
		d_neighbours.west.qy0[idx] = d_assem_sol.qy0[idx];
		d_neighbours.west.z0[idx]  = d_assem_sol.z0[idx];
		
		if (mwdg2)
		{
			if (test_case == 0)
			{
				d_neighbours.west.h1x[idx]  = C(0.0);
				d_neighbours.west.qx1x[idx] = C(0.0);
				d_neighbours.west.qy1x[idx] = C(0.0);
				d_neighbours.west.z1x[idx]  = C(0.0);
				
				d_neighbours.west.h1y[idx]  = C(0.0);
				d_neighbours.west.qx1y[idx] = C(0.0);
				d_neighbours.west.qy1y[idx] = C(0.0);
				d_neighbours.west.z1y[idx]  = C(0.0);
			}
			else
			{
				if (flow_NS)
				{
					d_neighbours.west.h1y[idx]  = d_assem_sol.h1y[idx];
					d_neighbours.west.qx1y[idx] = d_assem_sol.qx1y[idx];
					d_neighbours.west.qy1y[idx] = d_assem_sol.qy1y[idx];
					d_neighbours.west.z1y[idx]  = d_assem_sol.z1y[idx];

					d_neighbours.west.h1x[idx]  = C(0.0);
					d_neighbours.west.qx1x[idx] = C(0.0);
					d_neighbours.west.qy1x[idx] = C(0.0);
					d_neighbours.west.z1x[idx]  = C(0.0);
				}

				if ( flow_EW || (!flow_NS && !flow_EW) )
				{
					if (!flow_NS && !flow_EW)
					{
						d_assem_sol.h1x[idx]  = C(0.0);
						d_assem_sol.qx1x[idx] = C(0.0);
						d_assem_sol.qy1x[idx] = C(0.0);
						d_assem_sol.z1x[idx]  = C(0.0);
					}

					d_assem_sol.h1y[idx]  = C(0.0);
					d_assem_sol.qx1y[idx] = C(0.0);
					d_assem_sol.qy1y[idx] = C(0.0);
					d_assem_sol.z1y[idx]  = C(0.0);

					d_neighbours.west.h1x[idx]  = C(0.0);
					d_neighbours.west.qx1x[idx] = C(0.0);
					d_neighbours.west.qy1x[idx] = C(0.0);
					d_neighbours.west.z1x[idx]  = C(0.0);

					d_neighbours.west.h1y[idx]  = C(0.0);
					d_neighbours.west.qx1y[idx] = C(0.0);
					d_neighbours.west.qy1y[idx] = C(0.0);
					d_neighbours.west.z1y[idx]  = C(0.0);
				}
			}	
		}
		
		bound  = ( boundaries.west.bound(y) );
		border = ( x == 0 );

		finebdy = (finest && bound && border);

		if (finebdy && test_case == 0)
		{
			if (boundaries.west.bdytype == FREE)
			{
				d_neighbours.west.h0[idx]  = pow( (d_assem_sol.qx0[idx] * d_assem_sol.qx0[idx]) / sim_params.g, C(1.0) / C(3.0) );
				d_neighbours.west.qx0[idx] = d_assem_sol.qx0[idx]; 
				d_neighbours.west.qy0[idx] = d_assem_sol.qy0[idx];
			}
			else if
			(
				boundaries.west.bdytype == HFIX
				|| 
				boundaries.west.bdytype == HVAR
			)
			{
				if (sim_params.is_monai)
				{
					real hp  = d_assem_sol.h0[idx];
					real qxp = d_assem_sol.qx0[idx];
					real up  = (hp > solver_params.tol_h) ? qxp / hp : C(0.0);

					real hb = C(0.13535);
					real ub = C(0.0);

					twodoubles outputs = non_reflective_wave
					(
						boundaries.west.inlet, 
						dt, 
						dx_finest, 
						hp, up, hb, ub, 
						sim_params.g
					);

					d_neighbours.west.h0[idx]  = outputs.hb - d_assem_sol.z0[idx];
					d_neighbours.west.qx0[idx] = outputs.hb * outputs.ub;
				}
				else if (sim_params.is_oregon)
				{
				    d_neighbours.west.h0[idx] = d_assem_sol.h0[idx];
					
					// wavemaker speed based on bc2amr.f from https://zenodo.org/record/1419317
					real s = C(0.6) * exp( -C(0.25) * ( time_now - C(14.75) * ( time_now - C(14.75) ) );
					
					real v0 = (d_assem_sol.h0[idx] > solver_params.tol_h) ? d_assem_sol.qx0[idx] / d_assem_sol.h0[idx] : C(0.0);
					
					d_neighbours.west.qx0[idx] = ( (C(2.0) * s - v0) * d_neighbours.west.h0[idx] );
				}
				else
				{
					d_assem_sol.h0[idx]       = boundaries.west.inlet - d_assem_sol.z0[idx];
				    d_neighbours.west.h0[idx] = d_assem_sol.h0[idx];
				}
			}
			else if
			(
				boundaries.west.bdytype == QFIX
				|| 
				boundaries.west.bdytype == QVAR
			)
			{
				d_assem_sol.h0[idx]       = boundaries.west.q_src(dt, dx_finest);
				d_neighbours.west.h0[idx] = d_assem_sol.h0[idx];
			}
		}
	}
}