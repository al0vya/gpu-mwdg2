#pragma once

#include "../types/real.h"

#define H0  C(1.0) / sqrt( C(2.0) )
#define H1  C(1.0) / sqrt( C(2.0) )
#define G0  C(1.0) / sqrt( C(2.0) )
#define G1 -C(1.0) / sqrt( C(2.0) )

#define HH0_11 ( C(1.0) / C(2.0) ) 
#define HH0_12 ( C(0.0) )
#define HH0_13 ( C(0.0) )

#define HH0_21 ( -sqrt( C(3.0) ) / C(4.0) ) 
#define HH0_22 ( C(1.0) / C(4.0) ) 
#define HH0_23 ( C(0.0) )

#define HH0_31 ( -sqrt( C(3.0) ) / C(4.0) ) 
#define HH0_32 ( C(0.0) ) 
#define HH0_33 ( C(1.0) / C(4.0) ) 

#define HH1_11 ( C(1.0) / C(2.0) ) 
#define HH1_12 ( C(0.0) )
#define HH1_13 ( C(0.0) )

#define HH1_21 ( -sqrt( C(3.0) ) / C(4.0) ) 
#define HH1_22 ( C(1.0) / C(4.0) ) 
#define HH1_23 ( C(0.0) )

#define HH1_31 ( sqrt( C(3.0) ) / C(4.0) ) 
#define HH1_32 ( C(0.0) ) 
#define HH1_33 ( C(1.0) / C(4.0) ) 

#define HH2_11 ( C(1.0) / C(2.0) ) 
#define HH2_12 ( C(0.0) )
#define HH2_13 ( C(0.0) )

#define HH2_21 ( sqrt( C(3.0) ) / C(4.0) ) 
#define HH2_22 ( C(1.0) / C(4.0) ) 
#define HH2_23 ( C(0.0) )

#define HH2_31 ( -sqrt( C(3.0) ) / C(4.0) ) 
#define HH2_32 ( C(0.0) ) 
#define HH2_33 ( C(1.0) / C(4.0) ) 

#define HH3_11 ( C(1.0) / C(2.0) ) 
#define HH3_12 ( C(0.0) )
#define HH3_13 ( C(0.0) )

#define HH3_21 ( sqrt( C(3.0) ) / C(4.0) ) 
#define HH3_22 ( C(1.0) / C(4.0) ) 
#define HH3_23 ( C(0.0) )

#define HH3_31 ( sqrt( C(3.0) ) / C(4.0) ) 
#define HH3_32 ( C(0.0) ) 
#define HH3_33 ( C(1.0) / C(4.0) ) 

#define GA0_11 ( -sqrt( C(14.0) ) / C(14.0) )
#define GA0_12 ( -sqrt( C(42.0) ) / C(14.0) )
#define GA0_13 ( -sqrt( C(42.0) ) / C(14.0) )

#define GA0_21 ( C(0.0) ) 
#define GA0_22 ( C(0.0) )
#define GA0_23 ( -sqrt( C(2.0) ) / C(2.0) )

#define GA0_31 ( C(0.0) )
#define GA0_32 ( -sqrt( C(2.0) ) / C(2.0) )
#define GA0_33 ( C(0.0) )

#define GA1_11 ( C(0.0) ) 
#define GA1_12 ( C(0.0) ) 
#define GA1_13 ( C(0.0) ) 

#define GA1_21 ( C(0.0) ) 
#define GA1_22 ( C(0.0) )
#define GA1_23 ( C(0.0) ) 

#define GA1_31 ( C(0.0) )
#define GA1_32 ( C(0.0) ) 
#define GA1_33 ( C(0.0) )

#define GA2_11 ( C(0.0) ) 
#define GA2_12 ( C(0.0) ) 
#define GA2_13 ( C(0.0) ) 

#define GA2_21 ( C(0.0) ) 
#define GA2_22 ( C(0.0) )
#define GA2_23 ( C(0.0) ) 

#define GA2_31 ( C(0.0) )
#define GA2_32 ( C(0.0) ) 
#define GA2_33 ( C(0.0) )

#define GA3_11 ( sqrt( C(14.0) ) / C(14.0) )
#define GA3_12 ( -sqrt( C(42.0) ) / C(14.0) )
#define GA3_13 ( -sqrt( C(42.0) ) / C(14.0) )

#define GA3_21 ( C(0.0) ) 
#define GA3_22 ( C(0.0) )
#define GA3_23 ( sqrt( C(2.0) ) / C(2.0) )

#define GA3_31 ( C(0.0) )
#define GA3_32 ( sqrt( C(2.0) ) / C(2.0) )
#define GA3_33 ( C(0.0) )

#define GB0_11 ( C(0.0) ) 
#define GB0_12 ( C(0.0) ) 
#define GB0_13 ( C(0.0) ) 

#define GB0_21 ( C(0.0) ) 
#define GB0_22 ( C(0.0) )
#define GB0_23 ( C(0.0) ) 

#define GB0_31 ( C(0.0) )
#define GB0_32 ( C(0.0) ) 
#define GB0_33 ( C(0.0) )

#define GB1_11 ( -sqrt( C(14.0) ) / C(14.0) )
#define GB1_12 ( -sqrt( C(42.0) ) / C(14.0) )
#define GB1_13 ( sqrt( C(42.0) ) / C(14.0) )

#define GB1_21 ( C(0.0) ) 
#define GB1_22 ( C(0.0) )
#define GB1_23 ( -sqrt( C(2.0) ) / C(2.0) )

#define GB1_31 ( C(0.0) )
#define GB1_32 ( -sqrt( C(2.0) ) / C(2.0) )
#define GB1_33 ( C(0.0) )

#define GB2_11 ( sqrt( C(14.0) ) / C(14.0) )
#define GB2_12 ( -sqrt( C(42.0) ) / C(14.0) )
#define GB2_13 ( sqrt( C(42.0) ) / C(14.0) )

#define GB2_21 ( C(0.0) ) 
#define GB2_22 ( C(0.0) )
#define GB2_23 ( sqrt( C(2.0) ) / C(2.0) )

#define GB2_31 ( C(0.0) )
#define GB2_32 ( sqrt( C(2.0) ) / C(2.0) )
#define GB2_33 ( C(0.0) )

#define GB3_11 ( C(0.0) ) 
#define GB3_12 ( C(0.0) ) 
#define GB3_13 ( C(0.0) ) 

#define GB3_21 ( C(0.0) ) 
#define GB3_22 ( C(0.0) )
#define GB3_23 ( C(0.0) ) 

#define GB3_31 ( C(0.0) )
#define GB3_32 ( C(0.0) ) 
#define GB3_33 ( C(0.0) )

#define GC0_11 ( C(1.0) / C(2.0) ) 
#define GC0_12 ( C(0.0) ) 
#define GC0_13 ( C(0.0) ) 

#define GC0_21 ( -sqrt( C(21.0) ) / C(28.0) )
#define GC0_22 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )
#define GC0_23 ( C(2.0) * sqrt( C(7.0) ) / C(14.0) )

#define GC0_31 ( -sqrt( C(21.0) ) / C(28.0) )
#define GC0_32 ( C(2.0) * sqrt( C(7.0) ) / C(14.0) )
#define GC0_33 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )

#define GC1_11 ( -C(1.0) / C(2.0) ) 
#define GC1_12 ( C(0.0) ) 
#define GC1_13 ( C(0.0) ) 

#define GC1_21 ( -sqrt( C(21.0) ) / C(28.0) )
#define GC1_22 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )
#define GC1_23 ( -C(2.0) * sqrt( C(7.0) ) / C(14.0) )

#define GC1_31 ( sqrt( C(21.0) ) / C(28.0) )
#define GC1_32 ( -C(2.0) * sqrt( C(7.0) ) / C(14.0) )
#define GC1_33 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )

#define GC2_11 ( -C(1.0) / C(2.0) ) 
#define GC2_12 ( C(0.0) ) 
#define GC2_13 ( C(0.0) ) 

#define GC2_21 ( sqrt( C(21.0) ) / C(28.0) )
#define GC2_22 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )
#define GC2_23 ( -C(2.0) * sqrt( C(7.0) ) / C(14.0) )

#define GC2_31 ( -sqrt( C(21.0) ) / C(28.0) )
#define GC2_32 ( -C(2.0) * sqrt( C(7.0) ) / C(14.0) )
#define GC2_33 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )

#define GC3_11 ( C(1.0) / C(2.0) ) 
#define GC3_12 ( C(0.0) ) 
#define GC3_13 ( C(0.0) ) 

#define GC3_21 ( sqrt( C(21.0) ) / C(28.0) )
#define GC3_22 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )
#define GC3_23 ( C(2.0) * sqrt( C(7.0) ) / C(14.0) )

#define GC3_31 ( sqrt( C(21.0) ) / C(28.0) )
#define GC3_32 ( C(2.0) * sqrt( C(7.0) ) / C(14.0) )
#define GC3_33 ( -C(3.0) * sqrt( C(7.0) ) / C(28.0) )

/*#define H0     C(0.7071067811865475)
#define H1     C(0.7071067811865475)
#define G0     C(0.7071067811865475)
#define G1     C(-0.7071067811865475)

#define HH0_11 C(0.5)
#define HH0_12 C(0.0)
#define HH0_13 C(0.0)

#define HH0_21 C(-0.4330127018922193)
#define HH0_22 C(0.25)
#define HH0_23 C(0.0)

#define HH0_31 C(-0.4330127018922193)
#define HH0_32 C(0.0)
#define HH0_33 C(0.25)

#define HH1_11 C(0.5)
#define HH1_12 C(0.0)
#define HH1_13 C(0.0)

#define HH1_21 C(-0.4330127018922193)
#define HH1_22 C(0.25)
#define HH1_23 C(0.0)

#define HH1_31 C(0.4330127018922193)
#define HH1_32 C(0.0)
#define HH1_33 C(0.25)

#define HH2_11 C(0.5)
#define HH2_12 C(0.0)
#define HH2_13 C(0.0)

#define HH2_21 C(0.4330127018922193)
#define HH2_22 C(0.25)
#define HH2_23 C(0.0)

#define HH2_31 C(-0.4330127018922193)
#define HH2_32 C(0.0)
#define HH2_33 C(0.25)

#define HH3_11 C(0.5)
#define HH3_12 C(0.0)
#define HH3_13 C(0.0)

#define HH3_21 C(0.4330127018922193)
#define HH3_22 C(0.25)
#define HH3_23 C(0.0)

#define HH3_31 C(0.4330127018922193)
#define HH3_32 C(0.0)
#define HH3_33 C(0.25)

#define GA0_11 C(-0.2672612419124244)
#define GA0_12 C(-0.4629100498862757)
#define GA0_13 C(-0.4629100498862757)

#define GA0_21 C(0.0)
#define GA0_22 C(0.0)
#define GA0_23 C(-0.7071067811865476)

#define GA0_31 C(0.0)
#define GA0_32 C(-0.7071067811865476)
#define GA0_33 C(0.0)

#define GA1_11 C(0.0)
#define GA1_12 C(0.0)
#define GA1_13 C(0.0)

#define GA1_21 C(0.0)
#define GA1_22 C(0.0)
#define GA1_23 C(0.0)

#define GA1_31 C(0.0)
#define GA1_32 C(0.0)
#define GA1_33 C(0.0)

#define GA2_11 C(0.0)
#define GA2_12 C(0.0)
#define GA2_13 C(0.0)

#define GA2_21 C(0.0)
#define GA2_22 C(0.0)
#define GA2_23 C(0.0)

#define GA2_31 C(0.0)
#define GA2_32 C(0.0)
#define GA2_33 C(0.0)

#define GA3_11 C(0.2672612419124244)
#define GA3_12 C(-0.4629100498862757)
#define GA3_13 C(-0.4629100498862757)

#define GA3_21 C(0.0)
#define GA3_22 C(0.0)
#define GA3_23 C(0.7071067811865476)

#define GA3_31 C(0.0)
#define GA3_32 C(0.7071067811865476)
#define GA3_33 C(0.0)

#define GB0_11 C(0.0)
#define GB0_12 C(0.0)
#define GB0_13 C(0.0)

#define GB0_21 C(0.0)
#define GB0_22 C(0.0)
#define GB0_23 C(0.0)

#define GB0_31 C(0.0)
#define GB0_32 C(0.0)
#define GB0_33 C(0.0)

#define GB1_11 C(-0.2672612419124244)
#define GB1_12 C(-0.4629100498862757)
#define GB1_13 C(0.46291004988627577)

#define GB1_21 C(0.0)
#define GB1_22 C(0.0)
#define GB1_23 C(-0.7071067811865476)

#define GB1_31 C(0.0)
#define GB1_32 C(-0.7071067811865476)
#define GB1_33 C(0.0)

#define GB2_11 C(0.2672612419124244)
#define GB2_12 C(-0.4629100498862757)
#define GB2_13 C(0.46291004988627577)

#define GB2_21 C(0.0)
#define GB2_22 C(0.0)
#define GB2_23 C(0.7071067811865476)

#define GB2_31 C(0.0)
#define GB2_32 C(0.7071067811865476)
#define GB2_33 C(0.0)

#define GB3_11 C(0.0)
#define GB3_12 C(0.0)
#define GB3_13 C(0.0)

#define GB3_21 C(0.0)
#define GB3_22 C(0.0)
#define GB3_23 C(0.0)

#define GB3_31 C(0.0)
#define GB3_32 C(0.0)
#define GB3_33 C(0.0)

#define GC0_11 C(0.5)
#define GC0_12 C(0.0)
#define GC0_13 C(0.0)

#define GC0_21 C(-0.1636634176769942)
#define GC0_22 C(-0.2834733547569204)
#define GC0_23 C(0.37796447300922725)

#define GC0_31 C(-0.1636634176769942)
#define GC0_32 C(0.37796447300922725)
#define GC0_33 C(-0.2834733547569204)

#define GC1_11 C(-0.5)
#define GC1_12 C(0.0)
#define GC1_13 C(0.0)

#define GC1_21 C(-0.1636634176769942)
#define GC1_22 C(-0.2834733547569204)
#define GC1_23 C(-0.3779644730092272)

#define GC1_31 C(0.16366341767699427)
#define GC1_32 C(-0.3779644730092272)
#define GC1_33 C(-0.2834733547569204)

#define GC2_11 C(-0.5)
#define GC2_12 C(0.0)
#define GC2_13 C(0.0)

#define GC2_21 C(0.16366341767699427)
#define GC2_22 C(-0.2834733547569204)
#define GC2_23 C(-0.3779644730092272)

#define GC2_31 C(-0.1636634176769942)
#define GC2_32 C(-0.3779644730092272)
#define GC2_33 C(-0.2834733547569204)

#define GC3_11 C(0.5)
#define GC3_12 C(0.0)
#define GC3_13 C(0.0)

#define GC3_21 C(0.16366341767699427)
#define GC3_22 C(-0.2834733547569204)
#define GC3_23 C(0.37796447300922725)

#define GC3_31 C(0.16366341767699427)
#define GC3_32 C(0.37796447300922725)
#define GC3_33 C(-0.2834733547569204)*/