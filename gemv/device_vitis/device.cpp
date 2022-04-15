#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"
#include "xf_blas/helpers/dataMover/matMoverB2.hpp"

using namespace xf::blas;

typedef int t_TypeInt;
typedef double BLAS_dataType;

#define BLAS_parEntries 16
#define BLAS_logParEntries 4

extern "C" {
void gemm_kernel(double * p_a, double * p_x, double * p_y, double * result, unsigned int p_m, unsigned int p_n, BLAS_dataType p_alpha, BLAS_dataType p_beta) {
    #pragma HLS INTERFACE m_axi port=p_a offset=slave bundle=port1
    #pragma HLS INTERFACE m_axi port=p_x offset=slave bundle=port2
    #pragma HLS INTERFACE m_axi port=p_y offset=slave bundle=port3
    #pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=p_a bundle=control
    #pragma HLS INTERFACE s_axilite port=p_x bundle=control
    #pragma HLS INTERFACE s_axilite port=p_y bundle=control
    #pragma HLS INTERFACE s_axilite port=result bundle=control
    #pragma HLS INTERFACE s_axilite port=p_m bundle=control
    #pragma HLS INTERFACE s_axilite port=p_n bundle=control
    #pragma HLS INTERFACE s_axilite port=p_alpha bundle=control
    #pragma HLS INTERFACE s_axilite port=p_beta bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS DATAFLOW
    hls::stream<typename WideType<BLAS_dataType, 1 << BLAS_logParEntries>::t_TypeInt> l_strA;
    hls::stream<typename WideType<BLAS_dataType, 1 << BLAS_logParEntries>::t_TypeInt> l_strX;
    hls::stream<typename WideType<BLAS_dataType, 1>::t_TypeInt> l_strY;
    hls::stream<typename WideType<BLAS_dataType, 1>::t_TypeInt> l_strYR;
#pragma HLS DATAFLOW
    gem2Stream<BLAS_dataType, 1 <<BLAS_logParEntries>(p_m, p_n, p_a, l_strA);
    vec2GemStream<BLAS_dataType, 1 << BLAS_logParEntries>(p_m, p_n, p_x, l_strX);
    readVec2Stream<BLAS_dataType, 1>(p_y, p_m, l_strY);
    gemv<BLAS_dataType, BLAS_logParEntries>(p_m, p_n, p_alpha, l_strA, l_strX, p_beta, l_strY, l_strYR);
    writeStream2Vec<BLAS_dataType, 1>(l_strYR, p_m, result);
}

}
