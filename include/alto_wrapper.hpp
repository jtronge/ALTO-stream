#ifndef ALTO_WRAPPER_H
#define ALTO_WRAPPER_H

namespace alto {

class AltoTensorWrapper {
public:
    AltoTensorWrapper(int nmodes, uint64_t* dims, uint64_t nnz,
                      uint64_t** ind, double* vals, int nthreads);
    ~AltoTensorWrapper();
    void run_mttkrp(int mode, double** factor_mats, uint64_t rank);

private:
    struct Pimpl;
    std::unique_ptr<Pimpl> pimpl_;
};

}

#endif
