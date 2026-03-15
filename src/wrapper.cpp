#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>
#include <time.h>
#include <memory>

#include "poisson_generator.hpp"
#include "common.hpp"
#include "alto.hpp"
#include "cpd.hpp"

#include "streaming_cpd.hpp"
#include "constraints.hpp"

#include "alto_wrapper.hpp"

#include <unistd.h>
#include <sys/resource.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <climits>

namespace alto {

struct AltoTensorWrapper::Pimpl {
    AltoTensor<LIType>* at;
    std::vector<FType*> factor_ptrs;

    Pimpl(int nmodes, uint64_t* dims, uint64_t nnz, uint64_t** ind, double* vals, int nthreads)
    {
        if (sizeof(FType) != sizeof(double)) {
            throw std::runtime_error{"ALTO Ftype is not same as double type"};
        }

        SparseTensor* spt = AllocSparseTensor(nnz, nmodes);
        printf("generating alto sparse tensor\n");
        for (int m = 0; m < nmodes; ++m) {
            spt->dims[m] = dims[m];
            for (uint64_t n = 0; n < nnz; ++n) {
                spt->cidx[m][n] = ind[m][n];
            }
            factor_ptrs.push_back(nullptr);
        }
        for (uint64_t n = 0; n < nnz; ++n) {
            spt->vals[n] = vals[n];
        }

        // -1 set to skip vectorized streaming mode optimizations
        init_salto(spt, &at, nthreads, -1);
        // TODO: May want to check best number of partitions
        int num_partitions = get_num_ptrn(nthreads);
        update_salto(spt, at, num_partitions);
        DestroySparseTensor(spt);
    }

    ~Pimpl()
    {
        destroy_alto(at);
    }

    void run_mttkrp(int mode, double** factor_mats, uint64_t rank)
    {
        for (uint64_t m = 0; m < factor_ptrs.size(); ++m) {
            factor_ptrs[m] = factor_mats[m];
        }
        mttkrp_alto_atomic(mode, factor_ptrs.data(), at, factor_ptrs.size(), rank);
    }
};

AltoTensorWrapper::AltoTensorWrapper(int nmodes, uint64_t* dims, uint64_t nnz,
                                     uint64_t** ind, double* vals, int nthreads)
{
    pimpl_ = std::make_unique<AltoTensorWrapper::Pimpl>(nmodes, dims, nnz, ind, vals, nthreads);
}

AltoTensorWrapper::~AltoTensorWrapper()
{
}

void AltoTensorWrapper::run_mttkrp(int mode, double** factor_mats, uint64_t rank)
{
    pimpl_->run_mttkrp(mode, factor_mats, rank);
}

}
