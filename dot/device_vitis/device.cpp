#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"

using namespace xf::blas;

typedef int t_TypeInt;
typedef double BLAS_dataType;

#define BLAS_logParEntries 1

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

	  hls::stream<typename WideType<double, 1 << BLAS_logParEntries>::t_TypeInt> l_strX;
    hls::stream<typename WideType<double, 1 << BLAS_logParEntries>::t_TypeInt> l_strY;
#pragma HLS DATAFLOW
    readVec2Stream<double, 1 << BLAS_logParEntries>(x, n, l_strX);
    readVec2Stream<double, 1 << BLAS_logParEntries>(y, n, l_strY);
    dot<double, BLAS_logParEntries>(n, l_strX, l_strY, *result);
}

}
