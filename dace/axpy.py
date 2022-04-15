import argparse
import dace
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG

N = dace.symbol('N')


@dace.program(dace.float64[N], dace.float64[N])
def axpy(A: dace.float64[N], X: dace.float64[N]):
  b=0

  for i in dace.map[0:N]:
    b = b + (A[i] * X[i])

  return b


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    args = vars(parser.parse_args())

    print('Scalar-vector multiplication %d' % (args['N']))

    A = np.random.rand(args['N'])
    X = np.random.rand(args['N'])
    expected = A * X

    # Obtain SDFG from @dace.program
    sdfg = axpy.to_sdfg()

    # Convert SDFG to FPGA using a transformation
    sdfg.apply_transformations(FPGATransformSDFG)

    # Specialize and execute SDFG on FPGA
    sdfg._name = 'dot_fpga_%d' % args['N']
    sdfg.specialize(dict(N=args['N']))
    sdfg(A=A, X=X)
