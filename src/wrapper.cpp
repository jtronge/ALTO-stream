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

void make_alto_tensor(int nmodes, uint64_t* dims, uint64_t nnz, uint64_t** ind, double* vals, int nthreads)
{
    SparseTensor* spt = AllocSparseTensor(nnz, nmodes);
    printf("generating alto sparse tensor\n");
    for (int m = 0; m < nmodes; ++m) {
        spt->dims[m] = dims[m];
        for (uint64_t n = 0; n < nnz; ++n) {
            spt->cidx[m][n] = ind[m][n];
        }
    }
    for (uint64_t n = 0; n < nnz; ++n) {
        spt->vals[n] = vals[n];
    }

    AltoTensor<LIType>* at;

    // -1 set to skip vectorized streaming mode optimizations
    init_salto(spt, &at, nthreads, -1);
    // TODO: May want to check best number of partitions
    int num_partitions = get_num_ptrn(nthreads);
    update_salto(spt, at, num_partitions);
    DestroySparseTensor(spt);

    destroy_alto(at);
}

}
