#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"
#include <math.h>

extern "C" {
void l2norm_kernel(double * x, double * result, unsigned int n) {
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=port1
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=n bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    double accum=0;
    for (int i=0;i<n;i++) {
      double m=x[i]*x[i];
      accum+=m;
    }
    *result=sqrt(accum);
}

}
