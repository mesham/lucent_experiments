# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import argparse
import dace
import numpy as np
from dace.transformation.interstate import FPGATransformSDFG

N = dace.symbol('N')


@dace.program(dace.float64[N])
def l2norm(A: dace.float64[N]):
  accum=0

  for i in dace.map[0:N]:
    accum = accum + (A[i]**2)

  return dace.sqrt(accum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=24)
    args = vars(parser.parse_args())

    print('Scalar-vector multiplication %d' % (args['N']))

    A = np.random.rand(args['N'])
    expected = A

    # Obtain SDFG from @dace.program
    sdfg = l2norm.to_sdfg()

    # Convert SDFG to FPGA using a transformation
    sdfg.apply_transformations(FPGATransformSDFG)

    # Specialize and execute SDFG on FPGA
    sdfg._name = 'dot_fpga_%d' % args['N']
    sdfg.specialize(dict(N=args['N']))
    sdfg(A=A)
