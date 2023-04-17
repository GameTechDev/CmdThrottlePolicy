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

#include "D3D12MatMul.h"

void PrintUsage() {
    printf("Supported command line parameters:\n");
    printf(
        "--disable-command-throttle-policy-extension Don't use Command Throttle Policy Extension. "
        "By default we will set the command throttle policy to MAX_PERFORMANCE with Command "
        "Throttle Policy Extension.\n");
    printf(
        "--check-gpu-result Do matrix multiplication on CPU and compare the result with the one on "
        "GPU.\n");
    printf("-h Print helper information.\n");
}

int main(int argc, char* argv[]) {
    bool checkGPUResult = false;
    Settings settings = {};
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            PrintUsage();
            return 0;
        } else if (strcmp(argv[i], "--disable-command-throttle-policy-extension") == 0) {
            settings.disableCommandThrottlePolicyExtension = true;
        } else if (strcmp(argv[i], "--check-gpu-result") == 0) {
            checkGPUResult = true;
        } else {
            printf("Unsupported command line parameter: %s\n\n", argv[i]);
            PrintUsage();
            return 0;
        }
    }

    D3D12MatMul matMul(settings);

    matMul.DoMatMul();

    if (checkGPUResult) {
        matMul.CheckGPUResult();
    }

    return 0;
}
