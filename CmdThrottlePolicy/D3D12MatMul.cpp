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

#include <string>

#include <d3dcompiler.h>

#include "DXSampleHelper.h"

namespace {

ComPtr<ID3D12Resource> CreateBuffer(
    ID3D12Device* device,
    D3D12_HEAP_TYPE heapType,
    uint64_t size,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initialState) {
    D3D12_HEAP_PROPERTIES heapProperties = {};
    heapProperties.Type = heapType;
    heapProperties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProperties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProperties.CreationNodeMask = 0;
    heapProperties.VisibleNodeMask = 0;

    D3D12_RESOURCE_DESC bufferDescriptor = {};
    bufferDescriptor.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDescriptor.Alignment = 0;
    bufferDescriptor.Width = size;
    bufferDescriptor.Height = 1;
    bufferDescriptor.DepthOrArraySize = 1;
    bufferDescriptor.MipLevels = 1;
    bufferDescriptor.Format = DXGI_FORMAT_UNKNOWN;
    bufferDescriptor.SampleDesc.Count = 1;
    bufferDescriptor.SampleDesc.Quality = 0;
    bufferDescriptor.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDescriptor.Flags = flags;

    ComPtr<ID3D12Resource> buffer;
    device->CreateCommittedResource(
        &heapProperties, D3D12_HEAP_FLAG_NONE, &bufferDescriptor, initialState, nullptr,
        IID_PPV_ARGS(&buffer));
    return buffer;
}

void InitializeUploadBufferForInputBuffer(
    ID3D12Resource* uploadBuffer,
    uint64_t bufferSize,
    std::vector<float>* inputData) {
    uint64_t inputDataCount = bufferSize / sizeof(float);
    inputData->resize(inputDataCount);
    int sign = 1;
    for (uint64_t i = 0; i < inputDataCount; ++i) {
        (*inputData)[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    void* uploadPtr;
    ThrowIfFailed(uploadBuffer->Map(0, nullptr, &uploadPtr));
    memcpy(uploadPtr, inputData->data(), bufferSize);
    uploadBuffer->Unmap(0, nullptr);
}

void RecordResourceBarrier(
    ID3D12GraphicsCommandList* commandList,
    ID3D12Resource* resource,
    D3D12_RESOURCE_STATES Before,
    D3D12_RESOURCE_STATES After) {
    D3D12_RESOURCE_BARRIER barrierDesc = {};

    barrierDesc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrierDesc.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrierDesc.Transition.pResource = resource;
    barrierDesc.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrierDesc.Transition.StateBefore = Before;
    barrierDesc.Transition.StateAfter = After;

    commandList->ResourceBarrier(1, &barrierDesc);
}

struct ConstantBufferData {
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t TILE_K;
};

}  // anonymous namespace

D3D12MatMul::D3D12MatMul(const Settings& settings) {
    InitDevice();

    if (settings.disableCommandThrottlePolicyExtension || !InitIntelExtension()) {
        printf("The Command Throttle Policy Extension is disabled.\n\n");
    } else {
        printf("The Command Throttle Policy Extension is enabled.\n");
        printf(
            "You can disable the Command Throttle Policy Extension with "
            "--disable-command-throttle-policy-extension.\n\n");
    }

    InitQueue(settings);

    InitResources();
}

D3D12MatMul::~D3D12MatMul() {
    if (mINTCExtensionContext != nullptr) {
        HRESULT hr = INTC_DestroyDeviceExtensionContext(&mINTCExtensionContext);
        if (FAILED(hr)) {
            printf("\nERROR: INTC_DestroyDeviceExtensionContext failed.\n");
        } else {
            printf("\nSUCCESS: INTC_DestroyDeviceExtensionContext succeeded.\n");
        }
    }

    INTC_UnloadExtensionsLibrary();
}

void D3D12MatMul::InitDevice() {
    ComPtr<ID3D12Debug3> debugController;
    ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
    debugController->EnableDebugLayer();

    constexpr uint32_t kDXGIFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory2(kDXGIFactoryFlags, IID_PPV_ARGS(&factory)));

    ComPtr<IDXGIAdapter1> nonIntelAdapter;
    for (uint32_t adapterIndex = 0;
        DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(adapterIndex, &mHardwareAdapter);
        ++adapterIndex) {
        DXGI_ADAPTER_DESC1 adapterDescriptor;
        mHardwareAdapter->GetDesc1(&adapterDescriptor);
        if (adapterDescriptor.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            continue;
        }
        if (D3D12CreateDevice(
            mHardwareAdapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&mDevice)) != S_OK) {
            continue;
        }
        // Intel GPUs are preferred as currently the Command Throttle Policy Extension is only
        // available on Intel GPUs.
        if (adapterDescriptor.VendorId == 0x8086) {
            break;
        } else {
            nonIntelAdapter = mHardwareAdapter;
            mHardwareAdapter = nullptr;
        }
    }
    if (mHardwareAdapter.Get() == nullptr) {
        mHardwareAdapter = nonIntelAdapter;
    }

    PrintAdapterInfo();
}

bool D3D12MatMul::InitIntelExtension() {
    constexpr INTCExtensionVersion kRequiredVersion = { 1, 0, 0 }; //version 1.0.0

    if (SUCCEEDED(INTC_LoadExtensionsLibrary(false))) {
        printf("SUCCESS: INTC_LoadExtensionsLibrary succeeded.\n");
    } else {
        printf("ERROR: INTC_LoadExtensionsLibrary failed.\n");
        return false;
    }

    uint32_t supportedExtensionVersionCount = 0;
    if (SUCCEEDED(INTC_D3D12_GetSupportedVersions(mDevice.Get(), nullptr,
        &supportedExtensionVersionCount))) {
        printf("SUCCESS: INTC_D3D12_GetSupportedVersions 1 of 2 succeeded.\n");
    } else {
        printf("ERROR: INTC_D3D12_GetSupportedVersions 1 of 2 failed.\n");
        return false;
    }

    std::vector<INTCExtensionVersion> supportedExtensionVersions(supportedExtensionVersionCount);
    if (SUCCEEDED(INTC_D3D12_GetSupportedVersions(mDevice.Get(), supportedExtensionVersions.data(),
        &supportedExtensionVersionCount))) {
        printf("SUCCESS: INTC_D3D12_GetSupportedVersions 2 of 2 succeeded.\n");
    } else {
        printf("ERROR: INTC_D3D12_GetSupportedVersions 2 of 2 failed.\n");
        return false;
    }

    printf(
        "Locating requested extension version: %u.%u.%u...\n", kRequiredVersion.HWFeatureLevel,
        kRequiredVersion.APIVersion, kRequiredVersion.Revision);

    INTCExtensionInfo intcExtensionInfo = {};
    for (uint32_t i = 0; i < supportedExtensionVersionCount; ++i) {
        if ((supportedExtensionVersions[i].HWFeatureLevel >= kRequiredVersion.HWFeatureLevel) &&
            (supportedExtensionVersions[i].APIVersion >= kRequiredVersion.APIVersion) &&
            (supportedExtensionVersions[i].Revision >= kRequiredVersion.Revision)) {
            printf("SUCCESS: located requested version %u.%u.%u\n\n",
                supportedExtensionVersions[i].HWFeatureLevel,
                supportedExtensionVersions[i].APIVersion,
                supportedExtensionVersions[i].Revision);

            intcExtensionInfo.RequestedExtensionVersion = supportedExtensionVersions[i];
            break;
        } else {
            printf("%u.%u.%u doesn't match required version: %u.%u.%u, let's try the next one\n",
                supportedExtensionVersions[i].HWFeatureLevel,
                supportedExtensionVersions[i].APIVersion, supportedExtensionVersions[i].Revision,
                kRequiredVersion.HWFeatureLevel, kRequiredVersion.APIVersion,
                kRequiredVersion.Revision);
        }
    }

    if (SUCCEEDED(INTC_D3D12_CreateDeviceExtensionContext(mDevice.Get(), &mINTCExtensionContext,
        &intcExtensionInfo, nullptr))) {
        printf(
            "Let me tell you a little bit about this GPU:\n"
            "\tGPUMaxFrequency: %u Mhz\n"
            "\tGTGeneration: %u\n"
            "\tEUCount: %u\n"
            "\tPackageTDP: %u Watts\n"
            "\tMaxFillRate: %u pixels/clock@32bpp\n",
            intcExtensionInfo.IntelDeviceInfo.GPUMaxFreq,
            intcExtensionInfo.IntelDeviceInfo.GTGeneration,
            intcExtensionInfo.IntelDeviceInfo.EUCount, intcExtensionInfo.IntelDeviceInfo.PackageTDP,
            intcExtensionInfo.IntelDeviceInfo.MaxFillRate);
        printf("Done reporting intcExtensionInfo\n\n");
    } else {
        mINTCExtensionContext = nullptr;
        printf("ERROR: INTC_D3D12_CreateDeviceExtensionContext failed.\n");
        return false;
    }

    return true;
}

void D3D12MatMul::InitQueue(const Settings& settings) {
    D3D12_COMMAND_QUEUE_DESC queueDescriptor = {};
    queueDescriptor.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDescriptor.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    if (mINTCExtensionContext != nullptr) {
        // Create command queue with MAX_PERFORMANCE Command Throttle Policy
        INTC_D3D12_COMMAND_QUEUE_DESC intcQueueDescriptor = {};
        intcQueueDescriptor.pD3D12Desc = &queueDescriptor;
        intcQueueDescriptor.CommandThrottlePolicy =
            INTC_D3D12_COMMAND_QUEUE_THROTTLE_MAX_PERFORMANCE;
        ThrowIfFailed(INTC_D3D12_CreateCommandQueue(
            mINTCExtensionContext, &intcQueueDescriptor, IID_PPV_ARGS(&mQueue)));
    } else {
        ThrowIfFailed(mDevice->CreateCommandQueue(&queueDescriptor, IID_PPV_ARGS(&mQueue)));
    }

    ThrowIfFailed(mQueue->GetTimestampFrequency(&mTimestampFrequency));

    // Create objects for synchronization with mQueue.
    ThrowIfFailed(mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&mFence)));
    mFenceValue = 1;
    mFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (mFenceEvent == nullptr) {
        ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
    }
}

void D3D12MatMul::PrintAdapterInfo() {
    DXGI_ADAPTER_DESC1 adapterDescriptor;
    mHardwareAdapter->GetDesc1(&adapterDescriptor);
    wprintf(
        L"Device: %s (VendorID: 0x%04x DeviceID: 0x%04x)\n",
        adapterDescriptor.Description, adapterDescriptor.VendorId, adapterDescriptor.DeviceId);

    LARGE_INTEGER driverVersion;
    if (SUCCEEDED(mHardwareAdapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &driverVersion))) {
        uint64_t encoded = driverVersion.QuadPart;
        wprintf(
            L"Driver version: %d.%d.%d.%d\n", static_cast<uint16_t>((encoded >> 48) & 0xFFFF),
            static_cast<uint16_t>((encoded >> 32) & 0xFFFF),
            static_cast<uint16_t>((encoded >> 16) & 0xFFFF),
            static_cast<uint16_t>(encoded & 0xFFFF));
    }
    printf("\n");
}

void D3D12MatMul::InitResources() {
    CreateDescriptorHeap();
    CreateRootSignature();
    CreateComputePipeline();
    CreateBuffers();
    CreateBufferViews();
    CreateTimestampQueryHeap();
    CreateCommandList();

    InitBufferData();
}

void D3D12MatMul::CreateDescriptorHeap() {
    D3D12_DESCRIPTOR_HEAP_DESC heapDescriptor = {};
    // 1 CBV, 2 SRVs, 1 UAV
    heapDescriptor.NumDescriptors = 4;
    heapDescriptor.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    heapDescriptor.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    ThrowIfFailed(mDevice->CreateDescriptorHeap(&heapDescriptor, IID_PPV_ARGS(&mCBVSRVUAVHeap)));

    mCBVSRCUAVDescriptorSize =
        mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
}

void D3D12MatMul::CreateRootSignature() {
    D3D12_DESCRIPTOR_RANGE descriptorRanges[3];
    descriptorRanges[0].BaseShaderRegister = 0;
    descriptorRanges[0].NumDescriptors = 1;
    descriptorRanges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    descriptorRanges[0].RegisterSpace = 0;
    descriptorRanges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;

    descriptorRanges[1].BaseShaderRegister = 0;
    descriptorRanges[1].NumDescriptors = 2;
    descriptorRanges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    descriptorRanges[1].RegisterSpace = 0;
    descriptorRanges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;

    descriptorRanges[2].BaseShaderRegister = 0;
    descriptorRanges[2].NumDescriptors = 1;
    descriptorRanges[2].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    descriptorRanges[2].RegisterSpace = 0;
    descriptorRanges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;

    D3D12_ROOT_PARAMETER rootParameters[3];
    rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[0].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[0].DescriptorTable.pDescriptorRanges = &descriptorRanges[0];
    rootParameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[1].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[1].DescriptorTable.pDescriptorRanges = &descriptorRanges[1];
    rootParameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rootParameters[2].DescriptorTable.NumDescriptorRanges = 1;
    rootParameters[2].DescriptorTable.pDescriptorRanges = &descriptorRanges[2];

    D3D12_ROOT_SIGNATURE_DESC rootSignatureDescriptor = {};
    rootSignatureDescriptor.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    rootSignatureDescriptor.NumParameters = ARRAYSIZE(rootParameters);
    rootSignatureDescriptor.pParameters = rootParameters;
    rootSignatureDescriptor.NumStaticSamplers = 0;
    rootSignatureDescriptor.pStaticSamplers = nullptr;

    ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3D12SerializeRootSignature(
        &rootSignatureDescriptor, D3D_ROOT_SIGNATURE_VERSION_1_0, &mRootSignatureBlob, &error));
    ThrowIfFailed(mDevice->CreateRootSignature(
        0, mRootSignatureBlob->GetBufferPointer(), mRootSignatureBlob->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature)));
}

void D3D12MatMul::CreateComputePipeline() {
    ComPtr<ID3DBlob> computeShader;
    constexpr uint32_t kCompileFlags = 0;
    D3D_SHADER_MACRO defines[3];
    defines[0].Name = "LOCAL_GROUP_SIZE_X";
    std::string localGroupXStr = std::to_string(mLocalGroupSizeX);
    defines[0].Definition = localGroupXStr.c_str();
    defines[1].Name = "LOCAL_GROUP_SIZE_Y";
    std::string localGroupYStr = std::to_string(mLocalGroupSizeY);
    defines[1].Definition = localGroupYStr.c_str();
    defines[2] = {};
    ThrowIfFailed(D3DCompileFromFile(
        L"SLM_4X4_16X16_4_floats.hlsl", defines, nullptr, "main", "cs_5_0", kCompileFlags, 0,
        &computeShader, nullptr));

    D3D12_COMPUTE_PIPELINE_STATE_DESC computePipelineDescriptor = {};
    computePipelineDescriptor.pRootSignature = mRootSignature.Get();
    computePipelineDescriptor.NodeMask = 0;
    computePipelineDescriptor.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
    computePipelineDescriptor.CachedPSO.CachedBlobSizeInBytes = 0;
    computePipelineDescriptor.CachedPSO.pCachedBlob = nullptr;
    computePipelineDescriptor.CS.BytecodeLength = computeShader->GetBufferSize();
    computePipelineDescriptor.CS.pShaderBytecode = computeShader->GetBufferPointer();
    ThrowIfFailed(mDevice->CreateComputePipelineState(
        &computePipelineDescriptor, IID_PPV_ARGS(&mComputePipeline)));
}

void D3D12MatMul::CreateBuffers() {
    uint64_t constantBufferSize = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    mConstantBuffer = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_DEFAULT, constantBufferSize,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    uint64_t inputBufferSize1 = mM * mK * sizeof(float);
    mInputBuffer1 = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_DEFAULT, inputBufferSize1, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST);

    uint64_t inputBufferSize2 = mK * mN * sizeof(float);
    mInputBuffer2 = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_DEFAULT, inputBufferSize2, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST);

    uint64_t outputBufferSize = mM * mN * sizeof(float);
    mOutputBuffer = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_DEFAULT, outputBufferSize,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    uint64_t timestampsSize = 2 * sizeof(uint64_t);
    mTimestampBuffer = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_READBACK, timestampsSize, D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST);
}

void D3D12MatMul::CreateBufferViews() {
    uint64_t constantBufferSize = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
    D3D12_CPU_DESCRIPTOR_HANDLE heapStart = mCBVSRVUAVHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDescriptor = {};
    cbvDescriptor.BufferLocation = mConstantBuffer->GetGPUVirtualAddress();
    cbvDescriptor.SizeInBytes = static_cast<uint32_t>(constantBufferSize);
    mDevice->CreateConstantBufferView(&cbvDescriptor, heapStart);

    uint64_t inputElementsCount1 = mM * mK;
    D3D12_SHADER_RESOURCE_VIEW_DESC srvDescriptor = {};
    srvDescriptor.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDescriptor.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDescriptor.Format = DXGI_FORMAT_R32_TYPELESS;
    srvDescriptor.Buffer.FirstElement = 0;
    srvDescriptor.Buffer.NumElements = static_cast<uint32_t>(inputElementsCount1);
    srvDescriptor.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle1 = heapStart;
    srvHandle1.ptr += mCBVSRCUAVDescriptorSize;
    mDevice->CreateShaderResourceView(mInputBuffer1.Get(), &srvDescriptor, srvHandle1);

    uint64_t inputElementsCount2 = mK * mN;
    srvDescriptor.Buffer.NumElements = static_cast<uint32_t>(inputElementsCount2);
    D3D12_CPU_DESCRIPTOR_HANDLE srvHandle2 = heapStart;
    srvHandle2.ptr += mCBVSRCUAVDescriptorSize * 2;
    mDevice->CreateShaderResourceView(mInputBuffer2.Get(), &srvDescriptor, srvHandle2);

    uint64_t outputElementsCount = mM * mN;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDescriptor = {};
    uavDescriptor.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDescriptor.Format = DXGI_FORMAT_R32_TYPELESS;
    uavDescriptor.Buffer.FirstElement = 0;
    uavDescriptor.Buffer.CounterOffsetInBytes = 0;
    uavDescriptor.Buffer.NumElements = static_cast<uint32_t>(outputElementsCount);
    uavDescriptor.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    uavDescriptor.Buffer.StructureByteStride = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE uavHandle = heapStart;
    uavHandle.ptr += mCBVSRCUAVDescriptorSize * 3;
    mDevice->CreateUnorderedAccessView(mOutputBuffer.Get(), nullptr, &uavDescriptor, uavHandle);
}

void D3D12MatMul::CreateTimestampQueryHeap() {
    D3D12_QUERY_HEAP_DESC timestampHeapDesc = {};
    timestampHeapDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    timestampHeapDesc.Count = 2;
    ThrowIfFailed(mDevice->CreateQueryHeap(&timestampHeapDesc, IID_PPV_ARGS(&mTimestampQueryHeap)));
}

void D3D12MatMul::CreateCommandList() {
    ThrowIfFailed(mDevice->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&mCommandAllocator)));
    ThrowIfFailed(mDevice->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, mCommandAllocator.Get(), nullptr,
        IID_PPV_ARGS(&mCommandList)));
}

void D3D12MatMul::InitBufferData() {
    const uint64_t uploadBufferSize1 = mM * mK * sizeof(float);
    ComPtr<ID3D12Resource> uploadBuffer1 = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_UPLOAD, uploadBufferSize1,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    InitializeUploadBufferForInputBuffer(uploadBuffer1.Get(), uploadBufferSize1, &mInputData1);

    const uint64_t uploadBufferSize2 = mK * mN * sizeof(float);
    ComPtr<ID3D12Resource> uploadBuffer2 = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_UPLOAD, uploadBufferSize2,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    InitializeUploadBufferForInputBuffer(uploadBuffer2.Get(), uploadBufferSize2, &mInputData2);

    mCommandList->CopyBufferRegion(
        mInputBuffer1.Get(), 0, uploadBuffer1.Get(), 0, uploadBufferSize1);
    mCommandList->CopyBufferRegion(
        mInputBuffer2.Get(), 0, uploadBuffer2.Get(), 0, uploadBufferSize2);

    ComPtr<ID3D12Resource> uploadBufferForConstantBufferData = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_UPLOAD, sizeof(ConstantBufferData), D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ);
    void* uploadPtrForConstantBufferData = nullptr;
    ThrowIfFailed(uploadBufferForConstantBufferData->Map(
        0, nullptr, &uploadPtrForConstantBufferData));
    ConstantBufferData* constantBufferDataPtr =
        static_cast<ConstantBufferData*>(uploadPtrForConstantBufferData);
    constantBufferDataPtr->M = mM;
    constantBufferDataPtr->N = mN;
    constantBufferDataPtr->K = mK;
    constantBufferDataPtr->TILE_K = mTileK;
    uploadBufferForConstantBufferData->Unmap(0, nullptr);
    mCommandList->CopyBufferRegion(
        mConstantBuffer.Get(), 0, uploadBufferForConstantBufferData.Get(), 0,
        sizeof(ConstantBufferData));

    RecordResourceBarrier(
        mCommandList.Get(), mInputBuffer1.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    RecordResourceBarrier(
        mCommandList.Get(), mInputBuffer2.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    RecordResourceBarrier(
        mCommandList.Get(), mConstantBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

    mCommandList->Close();
    ID3D12CommandList* ppCommandLists[] = { mCommandList.Get() };
    mQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    WaitForGPUCompletion();
}

void D3D12MatMul::WaitForGPUCompletion() {
    ThrowIfFailed(mQueue->Signal(mFence.Get(), mFenceValue));

    ThrowIfFailed(mFence->SetEventOnCompletion(mFenceValue, mFenceEvent));
    WaitForSingleObjectEx(mFenceEvent, INFINITE, FALSE);

    ++mFenceValue;
}

void D3D12MatMul::DoMatMul() {
    constexpr int32_t kRowPerThread = 4;
    constexpr int32_t kColPerThread = 4;

    int32_t tileM = mLocalGroupSizeY * kRowPerThread;
    int32_t tileN = mLocalGroupSizeX * kColPerThread;
    int32_t dispatchX = static_cast<int32_t>(ceil(float(mN) / float(tileN)));
    int32_t dispatchY = static_cast<int32_t>(ceil(float(mM) / float(tileM)));
    printf(
        "M = %d, N = %d, K = %d, dispatchX = %d, dispatchY = %d\n\n", mM, mN, mK, dispatchX,
        dispatchY);

    ThrowIfFailed(mCommandList->Reset(mCommandAllocator.Get(), mComputePipeline.Get()));

    uint32_t beginTimestampIndex = 0;
    mCommandList->EndQuery(mTimestampQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);

    ID3D12DescriptorHeap* pHeaps[] = { mCBVSRVUAVHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(pHeaps), pHeaps);

    mCommandList->SetComputeRootSignature(mRootSignature.Get());

    D3D12_GPU_DESCRIPTOR_HANDLE cbvHandle = mCBVSRVUAVHeap->GetGPUDescriptorHandleForHeapStart();
    mCommandList->SetComputeRootDescriptorTable(0, cbvHandle);
    D3D12_GPU_DESCRIPTOR_HANDLE srvHandle = mCBVSRVUAVHeap->GetGPUDescriptorHandleForHeapStart();
    srvHandle.ptr += mCBVSRCUAVDescriptorSize;
    mCommandList->SetComputeRootDescriptorTable(1, srvHandle);
    D3D12_GPU_DESCRIPTOR_HANDLE uavHandle = mCBVSRVUAVHeap->GetGPUDescriptorHandleForHeapStart();
    uavHandle.ptr += 3 * mCBVSRCUAVDescriptorSize;
    mCommandList->SetComputeRootDescriptorTable(2, uavHandle);

    mCommandList->SetPipelineState(mComputePipeline.Get());
    mCommandList->Dispatch(dispatchX, dispatchY, 1);

    uint32_t endTimestampIndex = 1;
    mCommandList->EndQuery(
        mTimestampQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, endTimestampIndex);
    mCommandList->ResolveQueryData(
        mTimestampQueryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, beginTimestampIndex, 2,
        mTimestampBuffer.Get(), 0);

    ThrowIfFailed(mCommandList->Close());

    ID3D12CommandList* ppCommandLists[] = { mCommandList.Get() };
    mQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    WaitForGPUCompletion();

    void* pData = nullptr;
    ThrowIfFailed(mTimestampBuffer->Map(0, nullptr, &pData));
    const UINT64* pTimestamps = reinterpret_cast<UINT64*>(static_cast<UINT8*>(pData));

    const UINT64 gpuTimeUS = ((pTimestamps[1] - pTimestamps[0]) * 1000000) / mTimestampFrequency;
    printf("GPU execution time: %llu us\n\n", gpuTimeUS);

    mTimestampBuffer->Unmap(0, nullptr);
}

void D3D12MatMul::CheckGPUResult() {
    std::vector<float> outputDataCPU(mM * mN);
    std::vector<float> inputData1(mK);
    std::vector<float> inputData2(mK);
    printf("Do Matrix Multiplication on CPU.\n");
    printf("Total:\t\t");
    for (int32_t i = 0; i < mM / 100 + 1; ++i) {
        printf("-");
    }
    printf("\n");
    printf("Current:\t");
    for (int32_t y = 0; y < mM; ++y) {
        if (y % 100 == 0) {
            printf("-");
        }
        for (int32_t x = 0; x < mN; ++x) {
            for (int32_t x1 = 0; x1 < mK; ++x1) {
                inputData1[x1] = mInputData1[y * mK + x1];
            }
            for (int32_t x2 = 0; x2 < mK; ++x2) {
                inputData2[x2] = mInputData2[x2 * mK + x];
            }
            float output = 0;
            for (int32_t outputIndex = 0; outputIndex < mK; ++outputIndex) {
                output += inputData1[outputIndex] * inputData2[outputIndex];
            }
            outputDataCPU[y * mN + x] = output;
        }
    }
    printf("\nMatrix Multiplication on CPU is completed.\n");

    ThrowIfFailed(mCommandList->Reset(mCommandAllocator.Get(), nullptr));

    const uint64_t readbackBufferSize = mM * mN * sizeof(float);
    ComPtr<ID3D12Resource> readbackBuffer = CreateBuffer(
        mDevice.Get(), D3D12_HEAP_TYPE_READBACK, readbackBufferSize,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    RecordResourceBarrier(
        mCommandList.Get(), mOutputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COPY_SOURCE);
    mCommandList->CopyBufferRegion(
        readbackBuffer.Get(), 0, mOutputBuffer.Get(), 0, readbackBufferSize);
    RecordResourceBarrier(
        mCommandList.Get(), mOutputBuffer.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ThrowIfFailed(mCommandList->Close());

    ID3D12CommandList* ppCommandLists[] = { mCommandList.Get() };
    mQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
    WaitForGPUCompletion();

    void* pData = nullptr;
    ThrowIfFailed(readbackBuffer->Map(0, nullptr, &pData));
    const float* outputData = static_cast<const float*>(pData);

    // Currently we accept at most 3 ULP between CPU and GPU results.
    constexpr int32_t kToleranceULP = 3;
    constexpr int32_t kToleranceInBytes = 1 << kToleranceULP;
    printf(
        "Check the GPU result with the CPU result. Tolerance: %d ULPs\n", kToleranceULP);
    bool acceptGPUResult = true;
    for (int32_t y = 0; y < mM; ++y) {
        for (int32_t x = 0; x < mN; ++x) {
            int32_t outputDataGPUBytes = *reinterpret_cast<const int32_t*>(&outputData[y * mN + x]);
            int32_t outputDataCPUBytes =
                *reinterpret_cast<const int32_t*>(&outputDataCPU[y * mN + x]);
            if (abs(outputDataGPUBytes - outputDataCPUBytes) > kToleranceInBytes) {
                printf(
                    "At (%d, %d): GPU: %f CPU: %f\n", x, y, outputData[y * mN + x],
                    outputDataCPU[y * mN + x]);
                acceptGPUResult = false;
            }
        }
    }
    if (acceptGPUResult) {
        printf("\nThe GPU result is acceptable compared with the CPU result.\n");
    }
    readbackBuffer->Unmap(0, nullptr);
}
