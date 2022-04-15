#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"

using namespace xf::blas;

typedef int t_TypeInt;
typedef double BLAS_dataType;

#define BLAS_logParEntries 1

extern "C" {
void l2norm_kernel(double * x, double * result, unsigned int n) {
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=port1
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=n bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    double l_res;

    hls::stream<typename WideType<BLAS_dataType, 1 << BLAS_logParEntries>::t_TypeInt> l_strX;
#pragma HLS DATAFLOW
    readVec2Stream<BLAS_dataType, 1 << BLAS_logParEntries>(x, n, l_strX);
    nrm2<BLAS_dataType, BLAS_logParEntries>(n, l_strX, l_res);
    *result = l_res;
}

}
