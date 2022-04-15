#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"

using namespace xf::blas;


extern "C" {
void dot_kernel(double * x, double * y, double * result, unsigned int n) {
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=port1
    #pragma HLS INTERFACE m_axi port=y offset=slave bundle=port2
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=y bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=n bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    double acc=0;
    for (int i=0;i<n;i++) {
      double mult=x[i]*y[i];
      acc+=mult;
    }
    *result=acc;
}

}
