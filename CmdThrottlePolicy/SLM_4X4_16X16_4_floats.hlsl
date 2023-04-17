//*********************************************************
//
// Copyright 2023 Intel Corporation
//
// Permission is hereby granted, free of charge, to any
// person obtaining a copy of this software and associated
// documentation files(the "Software"), to deal in the Software
// without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the
// following conditions :
// The above copyright notice and this permission notice shall
// be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//*********************************************************

cbuffer ConstantBufferData : register(b0) {
    // inputMatrixA represents a M x K matrix, and inputMatrixB represents a K x N matrix.
    // outputMatrix represents a M x N matrix.
    int M;
    int K;
    int N;
    int TILE_K;
}

struct CS_INPUT {
    int3 localInvocationID : SV_GroupThreadID;
    int3 globalInvocationID : SV_DispatchThreadID;
};

ByteAddressBuffer inputMatrixA : register(t0);
ByteAddressBuffer inputMatrixB : register(t1);
RWByteAddressBuffer outputMatrix : register(u0);

// We ensure there won't be out-of-bound read or write.
float4 ReadFloat4FromA(int row, int col) {
    return asfloat(inputMatrixA.Load4(16 * (row * (K / 4) + col)));
}

float4 ReadFloat4FromB(int row, int col) {
    return asfloat(inputMatrixB.Load4(16 * (row * (N / 4) + col)));
}

void OutputFloat4(int row, int col, float4 value) {
    outputMatrix.Store4(16 * (row * (N / 4) + col), asuint(value));
}

// The shared memory to cache data from inputMatrixA and inputMatrixB.
groupshared float4 mm_Asub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X];
groupshared float4 mm_Bsub[LOCAL_GROUP_SIZE_Y * 4][LOCAL_GROUP_SIZE_X];

[numthreads(LOCAL_GROUP_SIZE_X, LOCAL_GROUP_SIZE_Y, 1)]
void main(CS_INPUT input) {
    // We always load or store a (4 x float4) together in each thread. So with all the threads
    // in one work group we can read (LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y * 16) items
    // from inputMatrixA and inputMatrixB into mm_Asub and mm_Bsub, and output
    // (LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y * 16) items into outputMatrix. We define each of
    // such (LOCAL_GROUP_SIZE_X * LOCAL_GROUP_SIZE_Y * 16) items as one tile.
    const int VEC_SIZE = 4;
    const int ROWS_PER_THREAD = 4;
    const int COLS_PER_THREAD = VEC_SIZE;

    // Get the indices of current thread in shared memory.
    int localRowIndex = input.localInvocationID.y * ROWS_PER_THREAD;
    int localColIndex = input.localInvocationID.x;

    // Get global indices of current thread.
    int globalRowIndex = input.globalInvocationID.y * ROWS_PER_THREAD;
    int globalColIndex = input.globalInvocationID.x;

    float4 acc[4];
    float4 ACached;
    float4 BCached[4];

    // Initialize acc with 0
    for (int innerRowIndexAcc = 0; innerRowIndexAcc < ROWS_PER_THREAD; ++innerRowIndexAcc) {
        acc[innerRowIndexAcc] = 0;
    }

    // Both inputMatrixA and inputMatrixB can be divided into multiple tiles.
    // We ensure K is a multiple of (LOCAL_GROUP_SIZE_X * VEC_SIZE).
    //
    //                  inputMatrixA                          inputMatrixB
    //    |                   K                  |     |            N            |
    //    |--------------------------------------|     |-------------------------|----
    //    |TileA1_1  TileA1_2 ... TileA1_numTiles|     |Tile  Tile      Tile     |tile
    //    |TileA2_1  TileA2_2 ... TileA2_numTiles|     |B1_1  B1_2  ... B1_j ... |Size
    // M  |  ...       ...    ...      ...       |  K  |-------------------------|----
    //    |TileAi_1  TileAi_2 ... TileAi_numTiles|     |Tile  Tile      Tile     |tile
    //    |  ...       ...    ...      ...       |     |B2_1  B2_2  ... B2_j ... |Size
    //    |--------  --------                    |     |-------------------------|----
    //    |tileSize  tileSize                    |     | ...   ...  ... ...      |
    //                                                 |-------------------------|
    //                                                 |Tile   ...  ... ...      |
    //                                                 |BnumTiles_1              |
    int tileSize = LOCAL_GROUP_SIZE_X * VEC_SIZE;
    int numTiles = K / tileSize;

    // The global column index of current tile in inputMatrixA.
    int globalColIndexA = localColIndex;
    // The global row index of current tile in inputMatrixB.
    int globalRowIndexB = input.localInvocationID.y * COLS_PER_THREAD;

    // The base column index in mm_Asub. Note that we don't need to multiply COLS_PER_THREAD to
    // tileColIndexA because each column represents a float4.
    int tileColIndexA = localColIndex;
    // The base row index in mm_Bsub.
    int tileRowIndexB = input.localInvocationID.y * COLS_PER_THREAD;
    for (int tileIndex = 0; tileIndex < numTiles; ++tileIndex) {
        // Load one tile of A into mm_Asub. In each thread, we load one (4 x float4) into mm_Asub.
        for (int innerRowIndexA = 0; innerRowIndexA < ROWS_PER_THREAD; ++innerRowIndexA) {
            int inputRow = localRowIndex + innerRowIndexA;
            int inputCol = tileColIndexA;

            mm_Asub[inputRow][inputCol] =
                ReadFloat4FromA(globalRowIndex + innerRowIndexA, globalColIndexA);
        }
        globalColIndexA += tileSize / VEC_SIZE;

        // Load one tile of B into mm_Bsub. In each thread, we load one (4 x float4) into mm_Bsub.
        for (int innerRowIndexB = 0; innerRowIndexB < COLS_PER_THREAD; ++innerRowIndexB) {
            int inputRow = tileRowIndexB + innerRowIndexB;
            int inputCol = localColIndex;

            mm_Bsub[inputRow][inputCol] =
                ReadFloat4FromB(globalRowIndexB + innerRowIndexB, globalColIndex);
        }
        globalRowIndexB += tileSize;

        // Ensure all the data for the current iteration has been loaded to mm_Asub and mm_Bsub.
        GroupMemoryBarrierWithGroupSync();

        // Compute acc (4 x float4) in a single thread.
        for (int mat4x4Index = 0; mat4x4Index < tileSize / VEC_SIZE; ++mat4x4Index) {
            // In each iteration we multiply two 4 x 4 matrices from mm_Asub and mm_Bsub.
            BCached[0] = mm_Bsub[mat4x4Index * VEC_SIZE][localColIndex];
            BCached[1] = mm_Bsub[mat4x4Index * VEC_SIZE + 1][localColIndex];
            BCached[2] = mm_Bsub[mat4x4Index * VEC_SIZE + 2][localColIndex];
            BCached[3] = mm_Bsub[mat4x4Index * VEC_SIZE + 3][localColIndex];

            ACached = mm_Asub[localRowIndex][mat4x4Index];
            acc[0] += BCached[0] * ACached.x;
            acc[0] += BCached[1] * ACached.y;
            acc[0] += BCached[2] * ACached.z;
            acc[0] += BCached[3] * ACached.w;

            ACached = mm_Asub[localRowIndex + 1][mat4x4Index];
            acc[1] += BCached[0] * ACached.x;
            acc[1] += BCached[1] * ACached.y;
            acc[1] += BCached[2] * ACached.z;
            acc[1] += BCached[3] * ACached.w;

            ACached = mm_Asub[localRowIndex + 2][mat4x4Index];
            acc[2] += BCached[0] * ACached.x;
            acc[2] += BCached[1] * ACached.y;
            acc[2] += BCached[2] * ACached.z;
            acc[2] += BCached[3] * ACached.w;

            ACached = mm_Asub[localRowIndex + 3][mat4x4Index];
            acc[3] += BCached[0] * ACached.x;
            acc[3] += BCached[1] * ACached.y;
            acc[3] += BCached[2] * ACached.z;
            acc[3] += BCached[3] * ACached.w;
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // Store the result (4 x float4) to outputMatrix.
    for (int innerRowIndex = 0; innerRowIndex < ROWS_PER_THREAD; ++innerRowIndex) {
        OutputFloat4(globalRowIndex + innerRowIndex, globalColIndex, acc[innerRowIndex]);
    }
}
