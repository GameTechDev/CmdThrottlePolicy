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

#ifndef D3D12_MAT_MUL_
#define D3D12_MAT_MUL_

#include <vector>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#define INTC_IGDEXT_D3D12
#include "igdext.h"

using Microsoft::WRL::ComPtr;

struct Settings {
    bool disableCommandThrottlePolicyExtension = false;
};

class D3D12MatMul {
public:
    explicit D3D12MatMul(const Settings& settings);

    // Destroy INTCExtensionContext and unload Intel extension library in the destructor
    ~D3D12MatMul();

    // Do a 1024x1024 matrix multiplication and print out the GPU execution time
    void DoMatMul();

    // Compare the result of the last GPU matrix multiplication with the one on CPU
    void CheckGPUResult();

private:
    // Initialize D3D12 resources
    void InitDevice();
    void InitQueue(const Settings& settings);

    void InitResources();
    void CreateDescriptorHeap();
    void CreateRootSignature();
    void CreateComputePipeline();
    void CreateBuffers();
    void CreateBufferViews();
    void CreateTimestampQueryHeap();
    void CreateCommandList();

    void InitBufferData();

    // Initialize Intel D3D12 extension
    bool InitIntelExtension();

    void WaitForGPUCompletion();

    void PrintAdapterInfo();

    ComPtr<IDXGIAdapter1> mHardwareAdapter;
    ComPtr<ID3D12Device> mDevice;

    HANDLE mFenceEvent;
    UINT64 mFenceValue;
    ComPtr<ID3D12Fence> mFence;
    ComPtr<ID3D12CommandQueue> mQueue;

    ComPtr<ID3D12CommandAllocator> mCommandAllocator;
    ComPtr<ID3D12GraphicsCommandList> mCommandList;

    ComPtr<ID3D12DescriptorHeap> mCBVSRVUAVHeap;
    uint32_t mCBVSRCUAVDescriptorSize;
    ComPtr<ID3DBlob> mRootSignatureBlob;
    ComPtr<ID3D12RootSignature> mRootSignature;
    ComPtr<ID3D12PipelineState> mComputePipeline;
    ComPtr<ID3D12Resource> mConstantBuffer;
    ComPtr<ID3D12Resource> mInputBuffer1;
    ComPtr<ID3D12Resource> mInputBuffer2;
    ComPtr<ID3D12Resource> mOutputBuffer;

    uint64_t mTimestampFrequency;
    ComPtr<ID3D12QueryHeap> mTimestampQueryHeap;
    ComPtr<ID3D12Resource> mTimestampBuffer;

    std::vector<float> mInputData1;
    std::vector<float> mInputData2;

    int32_t mLocalGroupSizeX = 16;
    int32_t mLocalGroupSizeY = 16;

    // Sizes of the matrix.
    // Input1: mM x mK Input2: mK x mN Output: mM x mN
    int32_t mM = 1024;
    int32_t mN = 1024;
    int32_t mK = 1024;
    int32_t mTileK = mLocalGroupSizeX * 4;

    // The pointer to an Intel D3D12 extension context.
    INTCExtensionContext* mINTCExtensionContext = nullptr;
};

#endif
