#include <stdio.h>
// 1. Include the LAPACKE header file.
//    This file is usually found in /usr/include or /usr/local/include
#include <lapacke.h>

int main(int argc, char *argv[]) {

    // Define the problem size
    lapack_int n = 2;     // The number of rows and columns in matrix A (2x2)
    lapack_int nrhs = 1;  // The number of right-hand side vectors (we have one vector b)

    // Define the matrix A and vector b
    // IMPORTANT: C stores 2D arrays in row-major order.
    // Matrix A = [[3, 7], [1, -4]] becomes a 1D array {3, 7, 1, -4}
    double a[4] = {3.0, 7.0, 1.0, -4.0};
    lapack_int lda = 2;   // The "leading dimension of A". For row-major, it's the number of columns.

    // Vector b = {1, 8}. After the call, this array will be overwritten with the solution vector x.
    double b[2] = {1.0, 8.0};
    lapack_int ldb = 1;   // The "leading dimension of B". For row-major, it's the number of columns in b (nrhs).

    // LAPACK will use this array to store pivot information.
    lapack_int ipiv[2];

    // Variable to hold the return value (info) from the LAPACK routine.
    // 0 indicates success.
    lapack_int info;

    printf("Starting LAPACKE dgesv solver...\n");
    printf("Matrix A:\n");
    printf("  %6.2f %6.2f\n", a[0], a[1]);
    printf("  %6.2f %6.2f\n", a[2], a[3]);
    printf("Vector b:\n");
    printf("  %6.2f\n", b[0]);
    printf("  %6.2f\n", b[1]);

    // 2. Call the LAPACKE routine to solve the system.
    info = LAPACKE_dgesv(
        LAPACK_ROW_MAJOR, // Tell LAPACKE we are using row-major storage.
        n,                // The order of the matrix A.
        nrhs,             // The number of right-hand sides.
        a,                // The matrix A. On exit, it's replaced by its LU factorization.
        lda,              // The leading dimension of A.
        ipiv,             // An array to be filled with pivot indices.
        b,                // The right-hand side vector b. On exit, it contains the solution x.
        ldb               // The leading dimension of b.
    );

    // 3. Check the result and print the solution.
    if (info > 0) {
        printf("The diagonal element of the U factor of A, U(%i,%i) is zero, so that A is singular;\n", info, info);
        printf("the solution could not be computed.\n");
        return 1;
    }

    printf("\nLAPACKE returned successfully (info = %d)\n", info);
    printf("Solution x:\n");
    printf("  %6.2f\n", b[0]);
    printf("  %6.2f\n", b[1]);
    // The correct answer is x = {4.0, -1.588...} -- wait, let me re-calculate that.
    // 3x + 7y = 1
    // 1x - 4y = 8  => x = 8 + 4y
    // 3(8+4y) + 7y = 1 => 24 + 12y + 7y = 1 => 19y = -23 => y = -23/19 ~= -1.21
    // x = 8 + 4(-23/19) = (152 - 92)/19 = 60/19 ~= 3.15
    // The correct solution is x = {3.15789, -1.21053}

    return 0;
}