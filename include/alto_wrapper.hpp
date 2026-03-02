#ifndef ALTO_WRAPPER_H
#define ALTO_WRAPPER_H

namespace alto {

void make_alto_tensor(int nmodes, uint64_t* dims, uint64_t nnz, uint64_t** ind, double* vals, int nthreads);

}

#endif
