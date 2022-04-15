#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"

using namespace xf::blas;

typedef int t_TypeInt;
typedef double BLAS_dataType;

#define BLAS_logParEntries 1

extern "C" {
void axpy_kernel(double * x, double * y, double * result, unsigned int n) {
    #pragma HLS INTERFACE m_axi port=x offset=slave bundle=port1
    #pragma HLS INTERFACE m_axi port=y offset=slave bundle=port2
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=x bundle=control
    #pragma HLS INTERFACE s_axilite port=y bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=n bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    double p_alpha=2.2;

	  hls::stream<typename WideType<BLAS_dataType, 1 << BLAS_logParEntries>::t_TypeInt> l_strX;
    hls::stream<typename WideType<BLAS_dataType, 1 << BLAS_logParEntries>::t_TypeInt> l_strY;
    hls::stream<typename WideType<BLAS_dataType, 1 << BLAS_logParEntries>::t_TypeInt> l_strR;
#pragma HLS DATAFLOW
    readVec2Stream<BLAS_dataType, 1 << BLAS_logParEntries>(x, n, l_strX);
    readVec2Stream<BLAS_dataType, 1 << BLAS_logParEntries>(y, n, l_strY);
    axpy<BLAS_dataType, 1 << BLAS_logParEntries>(n, p_alpha, l_strX, l_strY, l_strR);
    writeStream2Vec<BLAS_dataType, 1 << BLAS_logParEntries>(l_strR, n, result);
}

}
