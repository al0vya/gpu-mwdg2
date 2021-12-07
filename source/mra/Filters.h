#pragma once

#include "real.h"

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

/*
    printf("HH0_11: %f\n", HH0_11);
	printf("HH0_12: %f\n", HH0_12);
	printf("HH0_13: %f\n", HH0_13);
	printf("\n");
	printf("HH0_21: %f\n", HH0_21);
	printf("HH0_22: %f\n", HH0_22);
	printf("HH0_23: %f\n", HH0_23);
	printf("\n");
	printf("HH0_31: %f\n", HH0_31);
	printf("HH0_32: %f\n", HH0_32);
	printf("HH0_33: %f\n", HH0_33);
	printf("\n");
	printf("HH1_11: %f\n", HH1_11);
	printf("HH1_12: %f\n", HH1_12);
	printf("HH1_13: %f\n", HH1_13);
	printf("\n");
	printf("HH1_21: %f\n", HH1_21);
	printf("HH1_22: %f\n", HH1_22);
	printf("HH1_23: %f\n", HH1_23);
	printf("\n");
	printf("HH1_31: %f\n", HH1_31);
	printf("HH1_32: %f\n", HH1_32);
	printf("HH1_33: %f\n", HH1_33);
	printf("\n");
	printf("HH2_11: %f\n", HH2_11);
	printf("HH2_12: %f\n", HH2_12);
	printf("HH2_13: %f\n", HH2_13);
	printf("\n");
	printf("HH2_21: %f\n", HH2_21);
	printf("HH2_22: %f\n", HH2_22);
	printf("HH2_23: %f\n", HH2_23);
	printf("\n");
	printf("HH2_31: %f\n", HH2_31);
	printf("HH2_32: %f\n", HH2_32);
	printf("HH2_33: %f\n", HH2_33);
	printf("\n");
	printf("HH3_11: %f\n", HH3_11);
	printf("HH3_12: %f\n", HH3_12);
	printf("HH3_13: %f\n", HH3_13);
	printf("\n");
	printf("HH3_21: %f\n", HH3_21);
	printf("HH3_22: %f\n", HH3_22);
	printf("HH3_23: %f\n", HH3_23);
	printf("\n");
	printf("HH3_31: %f\n", HH3_31);
	printf("HH3_32: %f\n", HH3_32);
	printf("HH3_33: %f\n", HH3_33);
	printf("\n");
	printf("GA0_11: %f\n", GA0_11);
	printf("GA0_12: %f\n", GA0_12);
	printf("GA0_13: %f\n", GA0_13);
	printf("\n");
	printf("GA0_21: %f\n", GA0_21);
	printf("GA0_22: %f\n", GA0_22);
	printf("GA0_23: %f\n", GA0_23);
	printf("\n");
	printf("GA0_31: %f\n", GA0_31);
	printf("GA0_32: %f\n", GA0_32);
	printf("GA0_33: %f\n", GA0_33);
	printf("\n");
	printf("GA1_11: %f\n", GA1_11);
	printf("GA1_12: %f\n", GA1_12);
	printf("GA1_13: %f\n", GA1_13);
	printf("\n");
	printf("GA1_21: %f\n", GA1_21);
	printf("GA1_22: %f\n", GA1_22);
	printf("GA1_23: %f\n", GA1_23);
	printf("\n");
	printf("GA1_31: %f\n", GA1_31);
	printf("GA1_32: %f\n", GA1_32);
	printf("GA1_33: %f\n", GA1_33);
	printf("\n");
	printf("GA2_11: %f\n", GA2_11);
	printf("GA2_12: %f\n", GA2_12);
	printf("GA2_13: %f\n", GA2_13);
	printf("\n");
	printf("GA2_21: %f\n", GA2_21);
	printf("GA2_22: %f\n", GA2_22);
	printf("GA2_23: %f\n", GA2_23);
	printf("\n");
	printf("GA2_31: %f\n", GA2_31);
	printf("GA2_32: %f\n", GA2_32);
	printf("GA2_33: %f\n", GA2_33);
	printf("\n");
	printf("GA3_11: %f\n", GA3_11);
	printf("GA3_12: %f\n", GA3_12);
	printf("GA3_13: %f\n", GA3_13);
	printf("\n");
	printf("GA3_21: %f\n", GA3_21);
	printf("GA3_22: %f\n", GA3_22);
	printf("GA3_23: %f\n", GA3_23);
	printf("\n");
	printf("GA3_31: %f\n", GA3_31);
	printf("GA3_32: %f\n", GA3_32);
	printf("GA3_33: %f\n", GA3_33);
	printf("\n");
	printf("GB0_11: %f\n", GB0_11);
	printf("GB0_12: %f\n", GB0_12);
	printf("GB0_13: %f\n", GB0_13);
	printf("\n");
	printf("GB0_21: %f\n", GB0_21);
	printf("GB0_22: %f\n", GB0_22);
	printf("GB0_23: %f\n", GB0_23);
	printf("\n");
	printf("GB0_31: %f\n", GB0_31);
	printf("GB0_32: %f\n", GB0_32);
	printf("GB0_33: %f\n", GB0_33);
	printf("\n");
	printf("GB1_11: %f\n", GB1_11);
	printf("GB1_12: %f\n", GB1_12);
	printf("GB1_13: %f\n", GB1_13);
	printf("\n");
	printf("GB1_21: %f\n", GB1_21);
	printf("GB1_22: %f\n", GB1_22);
	printf("GB1_23: %f\n", GB1_23);
	printf("\n");
	printf("GB1_31: %f\n", GB1_31);
	printf("GB1_32: %f\n", GB1_32);
	printf("GB1_33: %f\n", GB1_33);
	printf("\n");
	printf("GB2_11: %f\n", GB2_11);
	printf("GB2_12: %f\n", GB2_12);
	printf("GB2_13: %f\n", GB2_13);
	printf("\n");printf("\n");
	printf("GB2_21: %f\n", GB2_21);
	printf("GB2_22: %f\n", GB2_22);
	printf("GB2_23: %f\n", GB2_23);
	printf("\n");
	printf("GB2_31: %f\n", GB2_31);
	printf("GB2_32: %f\n", GB2_32);
	printf("GB2_33: %f\n", GB2_33);
	printf("\n");
	printf("GB3_11: %f\n", GB3_11);
	printf("GB3_12: %f\n", GB3_12);
	printf("GB3_13: %f\n", GB3_13);
	printf("\n");
	printf("GB3_21: %f\n", GB3_21);
	printf("GB3_22: %f\n", GB3_22);
	printf("GB3_23: %f\n", GB3_23);
	printf("\n");
	printf("GB3_31: %f\n", GB3_31);
	printf("GB3_32: %f\n", GB3_32);
	printf("GB3_33: %f\n", GB3_33);
	printf("\n");
	printf("GC0_11: %f\n", GC0_11);
	printf("GC0_12: %f\n", GC0_12);
	printf("GC0_13: %f\n", GC0_13);
	printf("\n");
	printf("GC0_21: %f\n", GC0_21);
	printf("GC0_22: %f\n", GC0_22);
	printf("GC0_23: %f\n", GC0_23);
	printf("\n");
	printf("GC0_31: %f\n", GC0_31);
	printf("GC0_32: %f\n", GC0_32);
	printf("GC0_33: %f\n", GC0_33);
	printf("\n");
	printf("GC1_11: %f\n", GC1_11);
	printf("GC1_12: %f\n", GC1_12);
	printf("GC1_13: %f\n", GC1_13);
	printf("\n");
	printf("GC1_21: %f\n", GC1_21);
	printf("GC1_22: %f\n", GC1_22);
	printf("GC1_23: %f\n", GC1_23);
	printf("\n");
	printf("GC1_31: %f\n", GC1_31);
	printf("GC1_32: %f\n", GC1_32);
	printf("GC1_33: %f\n", GC1_33);
	printf("\n");
	printf("GC2_11: %f\n", GC2_11);
	printf("GC2_12: %f\n", GC2_12);
	printf("GC2_13: %f\n", GC2_13);
	printf("\n");
	printf("GC2_21: %f\n", GC2_21);
	printf("GC2_22: %f\n", GC2_22);
	printf("GC2_23: %f\n", GC2_23);
	printf("\n");
	printf("GC2_31: %f\n", GC2_31);
	printf("GC2_32: %f\n", GC2_32);
	printf("GC2_33: %f\n", GC2_33);
	printf("\n");
	printf("GC3_11: %f\n", GC3_11);
	printf("GC3_12: %f\n", GC3_12);
	printf("GC3_13: %f\n", GC3_13);
	printf("\n");
	printf("GC3_21: %f\n", GC3_21);
	printf("GC3_22: %f\n", GC3_22);
	printf("GC3_23: %f\n", GC3_23);
	printf("\n");
	printf("GC3_31: %f\n", GC3_31);
	printf("GC3_32: %f\n", GC3_32);
	printf("GC3_33: %f\n", GC3_33);*/

/*#define HH0_11 C(0.500000000000000)
#define HH0_12 C(0.000000000000000)
#define HH0_13 C(0.000000000000000)

#define HH0_21 C(-0.433012701892219)
#define HH0_22 C(0.250000000000000)
#define HH0_23 C(0.000000000000000)

#define HH0_31 C(-0.433012701892219)
#define HH0_32 C(0.000000000000000)
#define HH0_33 C(0.250000000000000)

#define HH1_11 C(0.500000000000000)
#define HH1_12 C(0.000000000000000)
#define HH1_13 C(0.000000000000000)

#define HH1_21 C(-0.433012701892219)
#define HH1_22 C(0.250000000000000)
#define HH1_23 C(0.000000000000000)

#define HH1_31 C(0.433012701892219)
#define HH1_32 C(0.000000000000000)
#define HH1_33 C(0.250000000000000)

#define HH2_11 C(0.500000000000000)
#define HH2_12 C(0.000000000000000)
#define HH2_13 C(0.000000000000000)

#define HH2_21 C(0.433012701892219)
#define HH2_22 C(0.250000000000000)
#define HH2_23 C(0.000000000000000)

#define HH2_31 C(-0.433012701892219)
#define HH2_32 C(0.000000000000000)
#define HH2_33 C(0.250000000000000)

#define HH3_11 C(0.500000000000000)
#define HH3_12 C(0.000000000000000)
#define HH3_13 C(0.000000000000000)

#define HH3_21 C(0.433012701892219)
#define HH3_22 C(0.250000000000000)
#define HH3_23 C(0.000000000000000)

#define HH3_31 C(0.433012701892219)
#define HH3_32 C(0.000000000000000)
#define HH3_33 C(0.250000000000000)

#define GA0_11 C(-0.267261241912424)
#define GA0_12 C(-0.462910049886276)
#define GA0_13 C(-0.462910049886276)

#define GA0_21 C(0.000000000000000)
#define GA0_22 C(0.000000000000000)
#define GA0_23 C(-0.707106781186548)

#define GA0_31 C(0.000000000000000)
#define GA0_32 C(-0.707106781186548)
#define GA0_33 C(0.000000000000000)

#define GA1_11 C(0.000000000000000)
#define GA1_12 C(0.000000000000000)
#define GA1_13 C(0.000000000000000)

#define GA1_21 C(0.000000000000000)
#define GA1_22 C(0.000000000000000)
#define GA1_23 C(0.000000000000000)

#define GA1_31 C(0.000000000000000)
#define GA1_32 C(0.000000000000000)
#define GA1_33 C(0.000000000000000)

#define GA2_11 C(0.000000000000000)
#define GA2_12 C(0.000000000000000)
#define GA2_13 C(0.000000000000000)

#define GA2_21 C(0.000000000000000)
#define GA2_22 C(0.000000000000000)
#define GA2_23 C(0.000000000000000)

#define GA2_31 C(0.000000000000000)
#define GA2_32 C(0.000000000000000)
#define GA2_33 C(0.000000000000000)

#define GA3_11 C(0.267261241912424)
#define GA3_12 C(-0.462910049886276)
#define GA3_13 C(-0.462910049886276)

#define GA3_21 C(0.000000000000000)
#define GA3_22 C(0.000000000000000)
#define GA3_23 C(0.707106781186548)

#define GA3_31 C(0.000000000000000)
#define GA3_32 C(0.707106781186548)
#define GA3_33 C(0.000000000000000)

#define GB0_11 C(0.000000000000000)
#define GB0_12 C(0.000000000000000)
#define GB0_13 C(0.000000000000000)

#define GB0_21 C(0.000000000000000)
#define GB0_22 C(0.000000000000000)
#define GB0_23 C(0.000000000000000)

#define GB0_31 C(0.000000000000000)
#define GB0_32 C(0.000000000000000)
#define GB0_33 C(0.000000000000000)

#define GB1_11 C(-0.267261241912424)
#define GB1_12 C(-0.462910049886276)
#define GB1_13 C(0.462910049886276)

#define GB1_21 C(0.000000000000000)
#define GB1_22 C(0.000000000000000)
#define GB1_23 C(-0.707106781186548)

#define GB1_31 C(0.000000000000000)
#define GB1_32 C(-0.707106781186548)
#define GB1_33 C(0.000000000000000)

#define GB2_11 C(0.267261241912424)
#define GB2_12 C(-0.462910049886276)
#define GB2_13 C(0.462910049886276)

#define GB2_21 C(0.000000000000000)
#define GB2_22 C(0.000000000000000)
#define GB2_23 C(0.707106781186548)

#define GB2_31 C(0.000000000000000)
#define GB2_32 C(0.707106781186548)
#define GB2_33 C(0.000000000000000)

#define GB3_11 C(0.000000000000000)
#define GB3_12 C(0.000000000000000)
#define GB3_13 C(0.000000000000000)

#define GB3_21 C(0.000000000000000)
#define GB3_22 C(0.000000000000000)
#define GB3_23 C(0.000000000000000)

#define GB3_31 C(0.000000000000000)
#define GB3_32 C(0.000000000000000)
#define GB3_33 C(0.000000000000000)

#define GC0_11 C(0.500000000000000)
#define GC0_12 C(0.000000000000000)
#define GC0_13 C(0.000000000000000)

#define GC0_21 C(-0.163663417676994)
#define GC0_22 C(-0.283473354756920)
#define GC0_23 C(0.377964473009227)

#define GC0_31 C(-0.163663417676994)
#define GC0_32 C(0.377964473009227)
#define GC0_33 C(-0.283473354756920)

#define GC1_11 C(-0.500000000000000)
#define GC1_12 C(0.000000000000000)
#define GC1_13 C(0.000000000000000)

#define GC1_21 C(-0.163663417676994)
#define GC1_22 C(-0.283473354756920)
#define GC1_23 C(-0.377964473009227)

#define GC1_31 C(0.163663417676994)
#define GC1_32 C(-0.377964473009227)
#define GC1_33 C(-0.283473354756920)

#define GC2_11 C(-0.500000000000000)
#define GC2_12 C(0.000000000000000)
#define GC2_13 C(0.000000000000000)

#define GC2_21 C(0.163663417676994)
#define GC2_22 C(-0.283473354756920)
#define GC2_23 C(-0.377964473009227)

#define GC2_31 C(-0.163663417676994)
#define GC2_32 C(-0.377964473009227)
#define GC2_33 C(-0.283473354756920)

#define GC3_11 C(0.500000000000000)
#define GC3_12 C(0.000000000000000)
#define GC3_13 C(0.000000000000000)

#define GC3_21 C(0.163663417676994)
#define GC3_22 C(-0.283473354756920)
#define GC3_23 C(0.377964473009227)

#define GC3_31 C(0.163663417676994)
#define GC3_32 C(0.377964473009227)
#define GC3_33 C(-0.283473354756920)*/