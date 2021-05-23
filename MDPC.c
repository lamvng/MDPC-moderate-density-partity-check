#include <stdio.h>
#include <stdlib.h>
#include <time.h> // To generate random number
#include <unistd.h>
#include <string.h>

/*
This file implements a structure of the matrix by dynamic allocation with pointers
*/



// Prange ISD
// Generate random matrix using pointers: 


// Function to swap two integer
void swap(int* x, int* y)
{
    int temp = *x;
    *x = *y;
    *y = temp;
}


// Input matrix from keyboard // For testing
void inputMatrix(int** mat, int size)
{
    printf("Input matrix from file:\n");
    FILE* fp = fopen("matrix.txt", "r");
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            fscanf(fp, "%d", &mat[i][j]);
        }
    }
}


void writeMatrix(int** mat, int size)
{
    printf("Input matrix from file:\n");
    FILE* fp = fopen("matrix.txt", "w");
    for (int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            fprintf(fp, "%2d", mat[i][j]);
        }
        fprintf(fp, "\n");
    }
}



void generateRandomBinaryMatrix(int** mat, int row, int col)
{
    for (int i=0;i<row;i++){
        for (int j=0;j<col;j++){
            mat[i][j] = rand() % 2; // Generate a random element
        }
    }
}


void generateRandomBinaryVector(int *vec, int size)
{
    for(int i=0; i<size; i++)
       vec[i] = rand() % 2;
}


// Print matrix
void printMatrix( int **mat, int size)
{
    // printf("\nMatrix dimensions: [%d,%d]\n", size, size);
    for (int i=0;i<size;i++){
        for (int j=0;j<size;j++){
            printf(" %2d ", mat[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


// Print vector
void printVector(int *vec, int size)
{
    for (int i=0;i<size;i++)
        printf("%2d", vec[i]);
    printf("\n");
}


// Generate identity matrix
void** generateIdentity (int size, int** mat)
{
    for(int i=0; i<size; i++) {
        for (int j=0; j<size; j++) {
            if (j==i)
                mat[i][j] = 1;
            else 
                mat[i][j] = 0;
        }
    }
}

// Find a tranpose of a square matrix
void tranpose(int** tranpose_mat, int** mat, int size)
{
	for (int i=0; i<size; i++)
	{
		for (int j=0; j<size; j++)
			tranpose_mat[i][j] = mat[j][i];
	}
}


// Multiply a vector and a matrix
void multiplyVectorMatrix(int* vec_result, int* vec, int** mat, int row_mat, int col_mat)
{
    for (int j=0; j<col_mat; j++){
        vec_result[j] = 0;
        for (int i=0; i<row_mat; i++)
            vec_result[j] += vec[i] * mat[i][j];
        vec_result[j] = vec_result[j] % 2;
    }
}


// Multiply a vector and a matrix (no modulo)
void multiplyVectorMatrixNoMod(int* vec_result, int* vec, int** mat, int row_mat, int col_mat)
{
    for (int j=0; j<col_mat; j++){
        vec_result[j] = 0;
        for (int i=0; i<row_mat; i++)
            vec_result[j] += vec[i] * mat[i][j];
    }
}


// Multiply a matrix and a vector
void multiplyMatrixVector(int* vec_result, int** mat, int* vec, int row_mat, int col_mat)
{
    for (int i=0; i<row_mat; i++){
        vec_result[i] = 0;
        for (int j=0; j<col_mat; j++)
            vec_result[i] += mat[i][j] * vec[j];
        vec_result[i] = vec_result[i] % 2;
    }
}



// Multiply two matrix
void multiplyMatrix(int** mat_result, int** mat1, int** mat2, int size)
{
    for (int i=0; i<size; i++){
        for (int j=0; j<size; j++){
            mat_result[i][j] = 0;
            for (int k=0; k<size; k++)
                mat_result[i][j] += mat1[i][k] * mat2[k][j];
            mat_result[i][j] = mat_result[i][j] % 2;
        }
    }
}

void vectorToBinary(int* vec, int size)
{
    for (int i=0; i<size; i++)
        vec[i] = vec[i] % 2;

}

// Calculate inversion by Gauss pivoting
// Return a flag_invertible
// flag_invertible = 0: Matrix is not inversible
// flag_invertible = 1: Matrix is inversible
int pivotGauss(int **mat, int **inv_mat, int size)
{
    int i, j, k;
    int row_to_swap;
    // int flag_invertible;
    int num_zero;
    // int division;

    // Loop on "pivot rows" i
    for (i=0; i<size; i++) {
        // If the diagonal element is zero, swap the row with another one
        if (mat[i][i] == 0) {
            row_to_swap = 1; // Offset of the new row to swap to
            // Find the row to swap
            while ((i+row_to_swap) < size && mat[i+row_to_swap][i] == 0)
                row_to_swap++;
            // Reach the last row without finding mat[i+row_to_swap][i] != 0
            // Break: Stop doing elementary operations
            if ((i+row_to_swap) == size)
                return 0; // The whole column is zero, return no-inversion

            // Swap row
            for (j = i, k = 0; k < size; k++) {
                swap(&mat[j][k], &mat[j+row_to_swap][k]);
                swap(&inv_mat[j][k], &inv_mat[j+row_to_swap][k]);

                }
        }

        // Doing element operation on every other row

        // // #NOTE: This is not necessary on binary
        // // On pivot row, make the diagonal =1
        // division = mat[i][i];
        // for (k=0; k<size; k++) {
        //     mat[i][k] = mat[i][k] / division;
        //     inv_mat[i][k] = inv_mat[i][k] / division;
        // }

        // Convert other rows
        for (j=0; j<size; j++) {
            // Skip the current "pivot row"
            if (i == j)
                continue;
            
            num_zero = 0;

            // Check if the whole row == 0
            if (mat[j][i] == 0) {
                for (k=0; k<size; k++) {
                    // if (k == i)
                    //     continue;
                    if (mat[j][k] == 0)
                        num_zero++;
                }
            }
            else {
                for (k=0; k<size; k++) {
                    // mat[j][i] = 1;  mat[i][i] = 1 --> Column i-th <-- (except on diagonal)
                    mat[j][k] = abs(mat[j][k] - mat[i][k]) % 2;
                    inv_mat[j][k] = abs(inv_mat[j][k] - inv_mat[i][k]) % 2;

                    if (mat[j][k] == 0)
                        num_zero++;
                }
            }
            
            // Check if the current row is all 0. If yes, then the matrix is not invertible
            if (num_zero == size)
                return 0;
        }
        // printf("\ni=%d\n",i);
        // printMatrix(mat, size);
        // printMatrix(inv_mat, size);
    }

    return 1;
}


int getHamming(int* vec, int size) {
    int weight = 0;
    int i;

    // vectorToBinary(vec, size);
    for (i=0; i<size; i++) {
        if (vec[i] == 1)
            weight++;
    }
    return weight;
}


void generateVectorwithWeight (int* vec, int size, int weight)
{
    int index_rand, i;
    int count;
    for (i=0; i<size; i++)
        vec[i] = 0;

    count = weight;

    while (count > 0) {
        index_rand = rand() % size;
        if (vec[index_rand] == 0) {
            vec[index_rand] = 1;
            count --;
        }
    }
}



// Calculate matrix circulant
// Test ok
void rot(int** mat_rot, int* vec, int size)
{
    int i, j;

    // First row
    for (j=0; j<size; j++)
            mat_rot[0][j] = vec[j];

    // Rotate the next rows
    for (i=1; i<size; i++) {
        mat_rot[i][0] = mat_rot[i-1][size-1];
        for (j=1; j<size; j++)
            mat_rot[i][j] = mat_rot[i-1][j-1];
    }
}


// Multiply vector by circulant matrix
void multiplyCirculant(int* result_vec, int* vec1, int* vec2, int size)
{
    int** vec2_rot = (int **) malloc(size * sizeof(int*));
    for(int i=0; i<size; i++)
        vec2_rot[i] = (int *) malloc(size * sizeof(int));

    rot(vec2_rot, vec2, size);
    multiplyVectorMatrix(result_vec, vec1, vec2_rot, size, size);

    free(vec2_rot);
}


// Generate private key h0 and h1
void generateKeys(int* h, int* h0, int* h1, int n, int w)
{
    int i, j;
    int flag_h0_invertible = 0;
    int** rot_h0, **inv_rot_h0;

    // Allocate space for rot_h0 and inv_rot_h0
    rot_h0 = (int **) malloc(n * sizeof(int*));
    inv_rot_h0 = (int **) malloc(n * sizeof(int*));
    for(i=0; i<n; i++) {
        rot_h0[i] = (int *) malloc(n * sizeof(int));
        inv_rot_h0[i] = (int *) malloc(n * sizeof(int));
    }

    // Generate h0 with weight w (private key) (until a invertible rot(h0) is found)
    while (flag_h0_invertible == 0) {
        generateVectorwithWeight(h0, n, w);

        // Calculate rot(h0)
        rot(rot_h0, h0, n);

        // Create and initiate the inversion matrix of A
        generateIdentity(n, inv_rot_h0);

        // Calculate the inversion of A, flag_invertible=1 if invertible
        flag_h0_invertible = pivotGauss(rot_h0, inv_rot_h0, n);
    }

    // Create h1 with weight w (private key)
    generateVectorwithWeight(h1, n, w);

    // Calculate public key h
    multiplyVectorMatrix(h, h1, inv_rot_h0, n, n);


    free(rot_h0);
    free(inv_rot_h0);
}



void concatenate(int* result_vec, int* vec1, int* vec2, int size)
{
    for (int i=0; i< size; i++) {
        result_vec[i] = vec1[i];
        result_vec[i+size] = vec2[i];
    }
}


// Write array (binary represenation) into binary file --> hash --> write to output file
// echo -n -e '\x66\x6f\x6f' | md5 > file.txt
void hash(int* vec)
{
    // Write vec to file
    FILE *write_ptr;
    write_ptr = fopen("input_hash.bin","wb");
    fwrite(vec,sizeof(vec),1,write_ptr);


    // Call system to calculate hash and write to file
    char command_calculate_hash[100];
    strcpy(command_calculate_hash, "cat input_hash.bin | sha256sum | awk '{ print $1 }' > output_hash.bin");
    system(command_calculate_hash);

    // The output has 65 hexa characters (Instead of 64 chars!). The last char (\x0a) is not read while getting the hash.
}



void encrypt(int* c0, int* c1, int* e0, int* e1, int* h, int* M, int n)
{
    int i, j;
    int* rot_e1h;
    int* conca_e0e1;
    int* hash_conca;

    // Concatenated string of e0 and e1
    conca_e0e1 = (int *) malloc(2*n * sizeof(int*));

    // Calculate hash of concatenated string (e0, e1)
    hash(conca_e0e1);

    // Read hash from file
    hash_conca = (int *) malloc(256 * sizeof(int*));
    FILE *ptr;
    ptr = fopen("output_hash.bin","rb");
    fread(hash_conca, 256, 1, ptr);

    // Calculate c0 = M xor hash_conca
    for (i=0; i<n; i++) {
        c0[i] = (M[i] + hash_conca[i]) % 2;
    }

    // Calculate c1 = e0 + e1*h (Multiplication by rotation matrix)
    rot_e1h = (int *) malloc(n * sizeof(int*));
    multiplyCirculant(rot_e1h, e1, h, n);
    for (i=0; i<n; i++) {
        c1[i] = e0[i] + rot_e1h[i];
    }
    free(rot_e1h);
    // free(ptr);
    free(conca_e0e1);
    free(hash_conca);
}

// Get u and v (or e1 and e2)
void gete0e1(int* e0, int* e1, int* uv, int n)
{
    for (int i=0; i<2*n; i++){
        e0[i] = uv[i];
        e1[i] = uv[i+n];
    }
}


int bitflip(int* e0, int* e1, int* h0, int* h1, int* c1, int T, int e, int n)
{
    int i, j;
    int* uv;
    int** H;
    int* s, *synd, *sum, *h_flipped, *h_uv, *s_huv;
    int* flipped_positions;
    int** roth0, **roth1, **roth0_t, **roth1_t;
    int w_u, w_v, w_s;

    // Initiate (u, v) of length 2n
    uv = (int *) malloc(2*n * sizeof(int*));
    for (i=0; i<2*n; i++)
        uv[i] = 0;


    // Calculate tranpose of rot(h0) and rot(h1)
    roth0 = (int **) malloc(n * sizeof(int*));
    roth1 = (int **) malloc(n * sizeof(int*));
    roth0_t = (int **) malloc(n * sizeof(int*));
    roth1_t = (int **) malloc(n * sizeof(int*));
    for(int i=0; i<n; i++) {
        roth0[i] = (int *) malloc(n * sizeof(int));
        roth1[i] = (int *) malloc(n * sizeof(int));
        roth0_t[i] = (int *) malloc(n * sizeof(int));
        roth1_t[i] = (int *) malloc(n * sizeof(int));
    }
    rot(roth0, h0, n);
    rot(roth1, h1, n);
    tranpose(roth0_t, roth0, n);
    tranpose(roth1_t, roth1, n);

    // Initiate H with dimension (n*2n) of rot(-h0).t and rot(h1).t
    H = (int **) malloc(n * sizeof(int*));
    for(i=0; i<n; i++) {
        H[i] = (int *) malloc(2*n * sizeof(int));
        for (j=0; j<n; j++) {
            H[i][j] = roth0[i][j];
            H[i][j+n] = roth1[i][j];
        }
    }


    // Initiate and calculate synd = h0 * rot(c1)
    s = (int *) malloc(n * sizeof(int*));
    multiplyCirculant(s, h0, c1, n);

    // Initiate synd <-- s
    synd = (int *) malloc(n * sizeof(int*));
    for (i=0; i<n; i++)
        synd[i] = s[i];

    // Initiate flipped_positions of length 2n
    flipped_positions = (int *) malloc(2*n * sizeof(int*));
    for (i=0; i<2*n; i++)
        flipped_positions[i] = 0;

    // Initiate vector "sum" of size 2n
    sum = (int *) malloc(2*n * sizeof(int*));

    // Initiate vector "h_flipped" of size n, = H * flipped_positions
    h_flipped = (int *) malloc(n * sizeof(int*));

    // Initiate vector "h_uv" of size n, = H * (u, v)
    h_uv = (int *) malloc(n * sizeof(int*));

    // Initiate vector "s_huv" of size n, = s - H*(u, v)
    s_huv = (int *) malloc(n * sizeof(int*));


    gete0e1(e0, e1, uv, n);
    while (((getHamming(e0, n) != e) || (getHamming(e1, n) != e)) && (getHamming(synd, n) > e)) {
        multiplyVectorMatrixNoMod(sum, synd, H, n, 2*n); // Multiplication without Mod

        // flipped_positions <-- 0
        for (i=0; i<2*n; i++)
            flipped_positions[i] = 0;

        for (i=0; i<2*n; i++) {
            // if sum[i] >= T
            if (sum[i] >= T)
                flipped_positions[i] = (flipped_positions[i] + 1) % 2;
        }

        // (u, v) = (u, v) xor flipped_positions
        for (i=0; i<2*n; i++)
            uv[i] = (uv[i] + flipped_positions[i]) % 2;

        // synd = synd - H * flipped_positions
        multiplyMatrixVector(h_flipped, H, flipped_positions, n, 2*n);
        for (i=0; i<n; i++)
            synd[i] = synd[i] - h_flipped[i];
    }

    // Calculate H * (u,v)
    multiplyMatrixVector(h_uv, H, uv, n, 2*n);

    // Calculate s - H*(u,v)
    for (i=0; i<n; i++){
        s_huv[i] = s[i] - h_uv[i];
    }

    // If hamming(s_huv) > e
    if (getHamming(s_huv, n) > e) {
        free(uv);
        free(H);
        free(synd);
        free(sum);
        free(roth0);
        free(roth1);
        free(roth0_t);
        free(roth1_t);
        free(h_uv);
        free(s_huv);

        return 0;
    }
    else {
        free(uv);
        free(H);
        free(synd);
        free(sum);
        free(roth0);
        free(roth1);
        free(roth0_t);
        free(roth1_t);
        free(h_uv);
        free(s_huv);
        
        // Get final e0 and e1
        for(i=0; i<n; i++) {
            e0[i] = uv[i];
            e1[i] = uv[i+n];
        }
        return 1;
    }
}


int decrypt(int* M_prime, int* c0, int* c1, int* e0, int* e1, int* h0, int* h1, int T, int e, int n)
{
    int i;
    int* conca_e0e1;
    int* hash_conca;
    int flag_bitflipping = 0;

    // Init seed for random()
    srand((unsigned int) time(NULL));
 

    // Get e0 and e1 by bitflipping
    flag_bitflipping = bitflip(e0, e1, h0, h1, c1, T, e, n);
    if (flag_bitflipping == 0)
        return 0;

    // Concatenated string of recovered e0 and e1
    conca_e0e1 = (int *) malloc(2*n * sizeof(int*));

    // Calculate hash of concatenated string (e0, e1)
    hash(conca_e0e1);


    // Read hash from file
    hash_conca = (int *) malloc(256 * sizeof(int*));
    FILE *ptr;
    ptr = fopen("output_hash.bin","rb");
    fread(hash_conca, 256, 1, ptr);


    // Calculate M_prime = c0 xor hash_conca
    for (i=0; i<n; i++) {
        M_prime[i] = (c0[i] + hash_conca[i]) % 2;
    }

    // free(ptr);
    free(conca_e0e1);
    free(hash_conca);

    return 1;
}




void main()
{
    int i, j;
    int *h0, *h1, *h;
    int *e0, *e1;
    int *c0, *c1;
    int *M, *M_prime; // Message
    int n; // Length of h0 and h1
    int w; // Weight of h0 and h1
    int e; // Error weight of e0 and e1
    int T;
    int flag_found = 0;

    // Param
    n = 300; // Length of h0 and h1
    w = 39; // Weight of h0 and h1
    e = 78;
    T = 26; // Seuil pour l'algo bitflip

    // Message
    int len_m = 50;
    M = (int *) malloc(len_m * sizeof(int*));
    M_prime = (int *) malloc(len_m * sizeof(int*));
    generateVectorwithWeight(M, len_m, 20);
    


    // Initiate h0, h1, and h
    h0 = (int *) malloc(n * sizeof(int*));
    h1 = (int *) malloc(n * sizeof(int*));
    h = (int *) malloc(n * sizeof(int*));

    // Initiate e0 and e1 with error weight e
    e0 = (int *) malloc(n * sizeof(int*));
    e1 = (int *) malloc(n * sizeof(int*));
    generateVectorwithWeight(e0, n, e);
    generateVectorwithWeight(e1, n, e);
    
    // Initiate c0 and c1
    c0 = (int *) malloc(n * sizeof(int*));
    c1 = (int *) malloc(n * sizeof(int*));

    // Generate keys
    generateKeys(h, h0, h1, n, w);

    printf("Private keys:\n");
    printf("h0:\n");
    for (i=0; i<n; i++)
        printf("%2d", h0[i]);
    printf("\n");

    printf("h1:\n");
    for (i=0; i<n; i++)
        printf("%2d", h1[i]);
    printf("\n");


    // Encrypt
    encrypt(c0, c1, e0, e1, h, M, n);

    // Print Results
    printf("\n\n\n\nParameters:\nn = %5d\nw = %5d\ne = %5d\n\n", n, w, e);

    printf("Message M:\n");
    for (i=0; i<len_m; i++) {
        printf("%2d", M[i]);
    }
    printf("\n\n");
    printf("\n\n");

    // Decrypt
    flag_found = decrypt(M_prime, c0, c1, e0, e1, h0, h1, T, e, n);
    if (flag_found == 0)
        printf("No e0 and e1 to recover.\n");
    else {
        // Print Results
        printf("Recovered M:\n");
        for (i=0; i<len_m; i++) {
            printf("%2d", M_prime[i]);
        }
        printf("\n\n");
    }
}