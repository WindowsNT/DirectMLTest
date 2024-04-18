// DirectMLTest.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <Windows.h>
#undef max
#undef min
#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <optional>

#include <d3d12.h>
#include "d3dx12.h"

#include <dxgi1_4.h>

#include <atlbase.h>
#include <combaseapi.h>

#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h> // The DirectML header from the Windows SDK.
#include "DirectMLX.h"

#pragma comment(lib,"d3d12.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"directml.lib")

void THROW_IF_FAILED(HRESULT hr)
{
    if (FAILED(hr)) throw;
}
class ML
{
public:
    CComPtr<ID3D12Device> d3D12Device;
    CComPtr<IDMLDevice> dmlDevice;
    CComPtr<ID3D12CommandQueue> commandQueue;
    CComPtr<ID3D12CommandAllocator> commandAllocator;
    CComPtr<ID3D12GraphicsCommandList> commandList;
    CComPtr<IDMLCommandRecorder> dmlCommandRecorder;
    CComPtr<IDMLCompiledOperator> dmlCompiledOperator;
    CComPtr<IDMLOperatorInitializer> dmlOperatorInitializer;
    DML_BINDING_PROPERTIES initializeBindingProperties = {}, executeBindingProperties = {};
    UINT descriptorCount = 0;
    CComPtr<ID3D12DescriptorHeap> descriptorHeap;


    HRESULT InitializeDirect3D12()
    {
        CComPtr<ID3D12Debug> d3D12Debug;

        // Throws if the D3D12 debug layer is missing - you must install the Graphics Tools optional feature
#if defined (_DEBUG)
        THROW_IF_FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&d3D12Debug)));
        d3D12Debug->EnableDebugLayer();
#endif

        CComPtr<IDXGIFactory4> dxgiFactory;
        CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));

        CComPtr<IDXGIAdapter> dxgiAdapter;
        UINT adapterIndex{};
        HRESULT hr{};
        do
        {
            dxgiAdapter = nullptr;
            dxgiAdapter = 0;
            THROW_IF_FAILED(dxgiFactory->EnumAdapters(adapterIndex, &dxgiAdapter));
            ++adapterIndex;

            d3D12Device = 0;
            hr = ::D3D12CreateDevice(
                dxgiAdapter,
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&d3D12Device));
            if (hr == DXGI_ERROR_UNSUPPORTED) continue;
            THROW_IF_FAILED(hr);
        } while (hr != S_OK);

        D3D12_COMMAND_QUEUE_DESC commandQueueDesc{};
        commandQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        commandQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;

        commandQueue = 0;
        THROW_IF_FAILED(d3D12Device->CreateCommandQueue(
            &commandQueueDesc,
            IID_PPV_ARGS(&commandQueue)));

        commandAllocator = 0;
        THROW_IF_FAILED(d3D12Device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&commandAllocator)));

        commandList = 0;
        THROW_IF_FAILED(d3D12Device->CreateCommandList(
            0,
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            commandAllocator,
            nullptr,
            IID_PPV_ARGS(&commandList)));

        return S_OK;
    }

    HRESULT CreateDML()
    {
        DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
#if defined (_DEBUG)
        dmlCreateDeviceFlags |= DML_CREATE_DEVICE_FLAG_DEBUG;
#endif
        return DMLCreateDevice(d3D12Device, dmlCreateDeviceFlags, IID_PPV_ARGS(&dmlDevice));
    }


    HRESULT CreateCommandRecorder()
    {
        return    dmlDevice->CreateCommandRecorder(
            IID_PPV_ARGS(&dmlCommandRecorder));
    }

    void CloseExecuteResetWait()
    {
        THROW_IF_FAILED(commandList->Close());

        ID3D12CommandList* commandLists[] = { commandList };
        commandQueue->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);

        CComPtr<ID3D12Fence> d3D12Fence;
        THROW_IF_FAILED(d3D12Device->CreateFence(
            0,
            D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&d3D12Fence)));

        auto hfenceEventHandle = ::CreateEvent(nullptr, true, false, nullptr);

        THROW_IF_FAILED(commandQueue->Signal(d3D12Fence, 1));
        THROW_IF_FAILED(d3D12Fence->SetEventOnCompletion(1, hfenceEventHandle));

        ::WaitForSingleObjectEx(hfenceEventHandle, INFINITE, FALSE);

        THROW_IF_FAILED(commandAllocator->Reset());
        THROW_IF_FAILED(commandList->Reset(commandAllocator, nullptr));
        CloseHandle(hfenceEventHandle);
    }

    auto CreateCompiledOperatorAbs(std::initializer_list<UINT32> j,UINT64* ts = 0)
    {
        dml::Graph graph(dmlDevice);
        dml::TensorDesc desc = { DML_TENSOR_DATA_TYPE_FLOAT32, j };
        dml::Expression input1 = dml::InputTensor(graph, 0, desc);
        dml::Expression output = dml::Abs(input1);

        if (ts)
            *ts = desc.totalTensorSizeInBytes;
        return graph.Compile(DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, { output });
    }


    auto CreateCompiledOperatorAdd(std::initializer_list<UINT32> j, UINT64* ts = 0)
    {
        dml::Graph graph(dmlDevice);

        auto desc1 = dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, j);
        auto input1 = dml::InputTensor(graph, 0, desc1);
        auto desc2 = dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, j);
        auto input2 = dml::InputTensor(graph, 1, desc2);

        auto output = dml::Add(input1,input2);
        if (ts)
            *ts = desc1.totalTensorSizeInBytes + desc2.totalTensorSizeInBytes;
        return graph.Compile(DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, { output });
    }

    // Two input tensors
    auto CreateCompiledOperatorLinearRegression(UINT32 N, UINT64* ts = 0)
    {
        dml::Graph graph(dmlDevice);

        auto desc1 = dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, { 1,N });
        auto desc2 = dml::TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, { 1,N });
        auto input1 = dml::InputTensor(graph, 0, desc1);
        auto input2 = dml::InputTensor(graph, 1, desc2);

        // Create first output tensor, calculate Sx by adding all first row of the tensor and going to the output tensor (in which , we will only take the last element as the sum)
        auto o1 = dml::CumulativeSummation(input1, 1, DML_AXIS_DIRECTION_INCREASING, false);

        // Sy, similarily
        auto o2 = dml::CumulativeSummation(input2, 1, DML_AXIS_DIRECTION_INCREASING, false);

        // xy, we calculate multiplication
        auto o3 = dml::Multiply(input1, input2);

        // Sxy
        auto o4 = dml::CumulativeSummation(o3, 1, DML_AXIS_DIRECTION_INCREASING, false);

        // x*x, we calculate multiplication
        auto o5 = dml::Multiply(input1, input1);

        // Sx2
        auto o6 = dml::CumulativeSummation(o5, 1, DML_AXIS_DIRECTION_INCREASING, false);

        auto d1 = desc1.totalTensorSizeInBytes;
        while (d1 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            d1++;
        auto d2 = desc2.totalTensorSizeInBytes;
        while (d2 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            d2++;

        if (ts)
            *ts = d1 + d2;
        return graph.Compile(DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION, { o1,o2,o3,o4,o5,o6 });
    }

    void CreateInitializer()
    {
        IDMLCompiledOperator* dmlCompiledOperators[] = { dmlCompiledOperator };
        THROW_IF_FAILED(dmlDevice->CreateOperatorInitializer(
            ARRAYSIZE(dmlCompiledOperators),
            dmlCompiledOperators,
            IID_PPV_ARGS(&dmlOperatorInitializer)));

    }

    void SetDescriptorHeaps()
    {
        ID3D12DescriptorHeap* d3D12DescriptorHeaps[] = { descriptorHeap };
        commandList->SetDescriptorHeaps(ARRAYSIZE(d3D12DescriptorHeaps), d3D12DescriptorHeaps);
    }

    void CreateHeap()
    {
        // You need to initialize an operator exactly once before it can be executed, and
        // the two stages require different numbers of descriptors for binding. For simplicity,
        // we create a single descriptor heap that's large enough to satisfy them both.
        initializeBindingProperties = dmlOperatorInitializer->GetBindingProperties();
        executeBindingProperties = dmlCompiledOperator->GetBindingProperties();
        descriptorCount = std::max(
            initializeBindingProperties.RequiredDescriptorCount,
            executeBindingProperties.RequiredDescriptorCount);

        // Create descriptor heaps.

        D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc{};
        descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        descriptorHeapDesc.NumDescriptors = descriptorCount;
        descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        THROW_IF_FAILED(d3D12Device->CreateDescriptorHeap(
            &descriptorHeapDesc,
            IID_PPV_ARGS(&descriptorHeap)));

        // Set the descriptor heap(s).
        SetDescriptorHeaps();
    }

    DML_BINDING_TABLE_DESC dmlBindingTableDesc{};
    CComPtr<IDMLBindingTable> dmlBindingTable;
    void CreateBindingTable()
    {
        dmlBindingTableDesc.Dispatchable = dmlOperatorInitializer;
        dmlBindingTableDesc.CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart();
        dmlBindingTableDesc.GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart();
        dmlBindingTableDesc.SizeInDescriptors = descriptorCount;

        THROW_IF_FAILED(dmlDevice->CreateBindingTable(
            &dmlBindingTableDesc,
            IID_PPV_ARGS(&dmlBindingTable)));

    }

    void ResetToExecute()
    {
        dmlBindingTableDesc.Dispatchable = dmlCompiledOperator;
        THROW_IF_FAILED(dmlBindingTable->Reset(&dmlBindingTableDesc));
    }

    CComPtr<ID3D12Resource> temporaryBuffer;
    UINT64 temporaryResourceSize = 0;
    void CreateTemporaryResources()
    {
        temporaryResourceSize = std::max(initializeBindingProperties.TemporaryResourceSize, executeBindingProperties.TemporaryResourceSize);
        if (temporaryResourceSize != 0)
        {
            auto x1 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto x2 = CD3DX12_RESOURCE_DESC::Buffer(temporaryResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
                &x1,
                D3D12_HEAP_FLAG_NONE,
                &x2,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&temporaryBuffer)));

            RebindTemporary();
        }
    }

    UINT64 persistentResourceSize = 0;
    CComPtr<ID3D12Resource> persistentBuffer;
    void CreatePersistentResources()
    {
        // Persistent sources
        persistentResourceSize = std::max(initializeBindingProperties.PersistentResourceSize, executeBindingProperties.PersistentResourceSize);
        if (persistentResourceSize != 0)
        {
            auto x1 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto x2 = CD3DX12_RESOURCE_DESC::Buffer(persistentResourceSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
                &x1,
                D3D12_HEAP_FLAG_NONE,
                &x2,
                D3D12_RESOURCE_STATE_COMMON,
                nullptr,
                IID_PPV_ARGS(&persistentBuffer)));

            RebindPersistent();
        }

    }

    void RebindPersistent()
    {
        // The persistent resource should be bound as the output to the IDMLOperatorInitializer.
        if (persistentResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{ persistentBuffer, 0, persistentResourceSize };
            DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
            dmlBindingTable->BindOutputs(1, &bindingDesc);
        }
    }

    void RebindTemporary()
    {
        if (temporaryResourceSize != 0)
        {
            DML_BUFFER_BINDING bufferBinding{ temporaryBuffer, 0, temporaryResourceSize };
            DML_BINDING_DESC bindingDesc{ DML_BINDING_TYPE_BUFFER, &bufferBinding };
            dmlBindingTable->BindTemporaryResource(&bindingDesc);
        }
    }


    CComPtr<ID3D12Resource> uploadBuffer;
    CComPtr<ID3D12Resource> inputBuffer;

    CComPtr<ID3D12Resource> outputBuffer;

    void CreateBuffers(size_t Total,void* data,std::vector<std::tuple<size_t,size_t>> ranges,bool O)
    {
        auto x1 = O ? CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT) : CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD);
        auto x2 = CD3DX12_RESOURCE_DESC::Buffer(Total,O ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE);
        THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
            &x1,
            D3D12_HEAP_FLAG_NONE,
            &x2,
            O ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(O ? &outputBuffer : &uploadBuffer)));
        if (O == 0)
        {
            auto x3 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT);
            auto x4 = CD3DX12_RESOURCE_DESC::Buffer(Total, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            THROW_IF_FAILED(d3D12Device->CreateCommittedResource(
                &x3,
                D3D12_HEAP_FLAG_NONE,
                &x4,
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS(&inputBuffer)));
        }

        if (O == 1)
        {
            std::vector<DML_BUFFER_BINDING> outputBufferBinding(ranges.size());
            std::vector<DML_BINDING_DESC> outputBindingDesc(ranges.size());
            for (size_t i = 0; i < ranges.size(); i++)
            {
                auto& ob = outputBufferBinding[i];
                ob.Buffer = outputBuffer;
                ob.Offset = std::get<0>(ranges[i]);
                ob.SizeInBytes = std::get<1>(ranges[i]);

                auto& od = outputBindingDesc[i];
                od.Type = DML_BINDING_TYPE_BUFFER;
                od.Desc = &ob;                    
            };
            dmlBindingTable->BindOutputs((UINT)ranges.size(), outputBindingDesc.data());
            return;
        }

        D3D12_SUBRESOURCE_DATA tensorSubresourceData{};
        tensorSubresourceData.pData = data;
        tensorSubresourceData.RowPitch = static_cast<LONG_PTR>(Total);
        tensorSubresourceData.SlicePitch = tensorSubresourceData.RowPitch;

        // Upload the input tensor to the GPU.
        ::UpdateSubresources(commandList, inputBuffer, uploadBuffer, 0, 0, 1, &tensorSubresourceData);

        auto x9 = CD3DX12_RESOURCE_BARRIER::Transition(inputBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        commandList->ResourceBarrier(1, &x9);

        std::vector<DML_BUFFER_BINDING> inputBufferBinding;
        inputBufferBinding.resize(ranges.size());
        std::vector<DML_BINDING_DESC> inputBindingDesc(ranges.size());
        for (size_t i = 0; i < ranges.size(); i++)
        {
            auto& ibb = inputBufferBinding[i];
            ibb.Buffer = inputBuffer;
            ibb.Offset = std::get<0>(ranges[i]);
            ibb.SizeInBytes = std::get<1>(ranges[i]);

            inputBindingDesc[i].Type = DML_BINDING_TYPE_BUFFER;
            inputBindingDesc[i].Desc = &ibb;

        }
        dmlBindingTable->BindInputs((UINT)inputBindingDesc.size(), inputBindingDesc.data());
    }

    void Record(int what)
    {
        if (what == 0)
            dmlCommandRecorder->RecordDispatch(commandList, dmlOperatorInitializer, dmlBindingTable);
        if (what == 1)
            dmlCommandRecorder->RecordDispatch(commandList, dmlCompiledOperator, dmlBindingTable);
    }
};


auto LinearRegressionCPU(float* px,float* py,size_t n)
{
    // Sx
    float Sx = 0, Sy = 0,Sxy = 0,Sx2 = 0;
    for (size_t i = 0; i < n; i++)
    {
        Sx += px[i];
        Sx2 += px[i] * px[i];
        Sy += py[i];
        Sxy += px[i] * py[i];
    }
    // B

    float B = (n * Sxy - Sx * Sy) / ((n * Sx2) - (Sx * Sx));
    float A = (Sy - (B * Sx)) / n;

    printf("Linear Regression CPU:\r\nSx = %f\r\nSy = %f\r\nSxy = %f\r\nSx2 = %f\r\nA = %f\r\nB = %f\r\n\r\n", Sx,Sy,Sxy,Sx2,A,B);
    return std::tuple<float,float>(A, B);
}


std::vector<float> xs = { 10,15,20,25,30,35 };
std::vector<float> ys = { 1003,1005,1010,1008,1014,1022 };
size_t N = xs.size();


#include <random>

void RandomData()
{
    xs.clear();
    ys.clear();
    
    int how = 1024 * 1024 * 32;

    printf("Generating %i random floats...",how);
    xs.resize(how);
    ys.resize(how);

    N = xs.size();
    std::random_device rd;
    std::mt19937 e2(rd());

    std::uniform_real_distribution<> dist1(0.0f, 1.0f);
    std::uniform_real_distribution<> dist2(10.0f, 100.0f);

    for (size_t i = 0; i < N; i++)
    {
        xs[i] = (float)dist1(e2);
        ys[i] = (float)dist2(e2);
    }


}

int main()
 {
 //   RandomData();


	CoInitialize(0);
    ML ml;

    // Initialize DirectX 12
    THROW_IF_FAILED(ml.InitializeDirect3D12());

    // Create the DirectML device.
    THROW_IF_FAILED(ml.CreateDML());

#define Method 3
#ifndef Method
#error Please define Method above to 1,2 or 3
#endif

    if (Method == 3)
        LinearRegressionCPU(xs.data(), ys.data(), N);

    UINT64 tensorInputSize = 0;
    std::vector<FLOAT> inputTensorElementArray;
    if (Method == 1)
    {
        // Compile the operators
        auto dmlc = ml.CreateCompiledOperatorAbs({ 2,2 }, &tensorInputSize);
        ml.dmlCompiledOperator.Attach(dmlc.Detach());
        inputTensorElementArray.resize(tensorInputSize / 4);
    }
    if (Method == 2)
    {
        // Compile the operators
        auto dmlc = ml.CreateCompiledOperatorAdd({ 2,2 }, &tensorInputSize);
        ml.dmlCompiledOperator.Attach(dmlc.Detach());
        inputTensorElementArray.resize(tensorInputSize /4);
    }
    if (Method == 3) // LR
    {
        auto dmlc = ml.CreateCompiledOperatorLinearRegression((UINT32)N, &tensorInputSize);
        ml.dmlCompiledOperator.Attach(dmlc.Detach());
        inputTensorElementArray.resize(tensorInputSize / 4);
    }
    // Initialize with numbers
    std::wstring inputString;
    if (Method != 3)
    {
        for (size_t i = 0; i < inputTensorElementArray.size(); i++)
        {
            inputTensorElementArray[i] = -1.0f * (i + 1);
            inputString += std::to_wstring(inputTensorElementArray[i]) + L' ';
        }
    }
    if (Method == 3)
    {
        // Initialize the linear regression
        inputString = L"";
        auto sz2 = inputTensorElementArray.size() / 2;
        memcpy(inputTensorElementArray.data(), xs.data(), std::min(sz2, xs.size()) * sizeof(float));
        memcpy(inputTensorElementArray.data() + sz2, ys.data(), std::min(sz2, ys.size()) * sizeof(float));
    }
    if (N <= 5)
        std::wcout << L"Inputs\r\n-------------\r\n" << inputString << std::endl << std::endl;

    auto tensorOutputSize = tensorInputSize;

    UINT64 Method3TensorSizes[6] = {};


    if (Method == 3)
    {
        // We need 6 tensors for intermediates and output

        // First output should be N*4 bytes, aligned
        UINT64 f1 = N * 4;
        while (f1 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            f1++;
        Method3TensorSizes[0] = f1;

        // Second output, same
        UINT64 f2 = N * 4;
        while (f2 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            f2++;
        Method3TensorSizes[1] = f2;

        // Third output, same
        UINT64 f3 = N * 4;
        while (f3 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            f3++;
        Method3TensorSizes[2] = f3;

        // Fourth output, same
        UINT64 f4 = N * 4;
        while (f4 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            f4++;
        Method3TensorSizes[3] = f4;

        // Fifth output, same
        UINT64 f5 = N * 4;
        while (f5 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            f5++;
        Method3TensorSizes[4] = f5;


        // Sixth output, same
        UINT64 f6 = N * 4;
        while (f6 % DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT)
            f6++;
        Method3TensorSizes[5] = f6;

        tensorOutputSize = (f1 + f2 + f3 + f4 + f5 + f6);
    }

    // Initialize
    ml.CreateInitializer();

    // https://learn.microsoft.com/en-us/windows/ai/directml/dml-binding
    // Query the operator for the required size (in descriptors) of its binding table.
    ml.CreateHeap();

    // Create a binding table over the descriptor heap we just created.
    ml.CreateBindingTable();

   // The temporary resource is scratch memory (used internally by DirectML), whose contents you don't need to define.
    ml.CreateTemporaryResources();

    // The persistent resource is long-lived, and you need to initialize it using the IDMLOperatorInitializer.
    ml.CreatePersistentResources();

    // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
    ml.CreateCommandRecorder();

    // Record execution of the operator initializer.
    ml.Record(0);

    // Close the Direct3D 12 command list, and submit it for execution as you would any other command list. You could
    // in principle record the execution into the same command list as the initialization, but you need only to Initialize
    // once, and typically you want to Execute an operator more frequently than that.
    ml.CloseExecuteResetWait();

    // Bind and execute the operator on the GPU.
    ml.SetDescriptorHeaps();

    // Reset the binding table to bind for the operator we want to execute (it was previously used to bind for the initializer).
    ml.ResetToExecute();

    // Rebind Temporary and Persistent stuff
    ml.RebindTemporary();
    ml.RebindPersistent();


   if (Method == 1)
       ml.CreateBuffers(tensorInputSize, inputTensorElementArray.data(), { {0,tensorInputSize} },0);

   if (Method == 2)
       ml.CreateBuffers(tensorInputSize, inputTensorElementArray.data(), { {0,tensorInputSize/2},{tensorInputSize/2,tensorInputSize/2} },0);

   if (Method == 3)
       ml.CreateBuffers(tensorInputSize, inputTensorElementArray.data(), { {0,tensorInputSize / 2},{tensorInputSize / 2,tensorInputSize / 2} },0);

    if (Method == 1)
        ml.CreateBuffers(tensorOutputSize, 0, { {0,tensorOutputSize} }, 1);

    if (Method == 2)
        ml.CreateBuffers(tensorOutputSize, 0, { {0,tensorOutputSize/2} }, 1);

    if (Method == 3)
        ml.CreateBuffers(tensorOutputSize, 0, { 
            {0,Method3TensorSizes[0]},
            {Method3TensorSizes[0],Method3TensorSizes[1]},
            {Method3TensorSizes[0] + Method3TensorSizes[1],Method3TensorSizes[2]},
            {Method3TensorSizes[0] + Method3TensorSizes[1] + Method3TensorSizes[2],Method3TensorSizes[3]},
            {Method3TensorSizes[0] + Method3TensorSizes[1] + Method3TensorSizes[2] + Method3TensorSizes[3],Method3TensorSizes[4]},
            {Method3TensorSizes[0] + Method3TensorSizes[1] + Method3TensorSizes[2] + Method3TensorSizes[3] + Method3TensorSizes[4],Method3TensorSizes[5]},
            }, 1);

    // Run it
    ml.Record(1);
    ml.CloseExecuteResetWait();
 
    // The output buffer now contains the result of the identity operator,
    // so read it back if you want the CPU to access it.
    CComPtr<ID3D12Resource> readbackBuffer;
    auto x7 = CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK);
    auto x8 = CD3DX12_RESOURCE_DESC::Buffer(tensorOutputSize);
    THROW_IF_FAILED(ml.d3D12Device->CreateCommittedResource(
        &x7,
        D3D12_HEAP_FLAG_NONE,
        &x8,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)));

    auto x10 = CD3DX12_RESOURCE_BARRIER::Transition(ml.outputBuffer,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    ml.commandList->ResourceBarrier(1,&x10);

    ml.commandList->CopyResource(readbackBuffer, ml.outputBuffer);

    ml.CloseExecuteResetWait();

    D3D12_RANGE tensorBufferRange{ 0, static_cast<SIZE_T>(tensorOutputSize) };
    FLOAT* outputBufferData{};
    THROW_IF_FAILED(readbackBuffer->Map(0, &tensorBufferRange, reinterpret_cast<void**>(&outputBufferData)));
    std::wstring outputString;

    float Sx = 0, Sy = 0, Sxy = 0, Sx2 = 0;
    if (Method == 1)
    {
        outputString += L"Output\r\n-----------------\r\n";
        for (size_t tensorElementIndex = 0; tensorElementIndex < 4 ; ++tensorElementIndex, ++outputBufferData)
            outputString += std::to_wstring(*outputBufferData) + L' ';
        outputString += L"\r\n\r\n";
        std::wcout << outputString;
    }
    if (Method == 2)
    {
        outputString += L"Output\r\n-----------------\r\n";
        for (size_t tensorElementIndex = 0; tensorElementIndex < 8; ++tensorElementIndex, ++outputBufferData)
            outputString += std::to_wstring(*outputBufferData) + L' ';
        outputString += L"\r\n\r\n";
        std::wcout << outputString;
    }
    if (Method == 3)
    {
        // Output 1, 
        char* o = (char*)outputBufferData;
        Sx = outputBufferData[N - 1];

        if (N <= 5)
        {
            outputString += L"Output 1 - Sx\r\n-----------------\r\n";
            for (size_t tensorElementIndex = 0; tensorElementIndex < N; ++tensorElementIndex, ++outputBufferData)
                outputString += std::to_wstring(*outputBufferData) + L' ';
            outputString += L"\r\n\r\n";
        }

        o += Method3TensorSizes[0];
        outputBufferData = (float*)o;
        Sy = outputBufferData[N - 1];
        if (N <= 5)
        {
            outputString += L"Output 2 - Sy\r\n-----------------\r\n";
            for (size_t tensorElementIndex = 0; tensorElementIndex < N; ++tensorElementIndex, ++outputBufferData)
                outputString += std::to_wstring(*outputBufferData) + L' ';
            outputString += L"\r\n\r\n";
        }

        o += Method3TensorSizes[1];
        outputBufferData = (float*)o;
        if (N <= 5)
        {
            outputString += L"Output 3 - xy\r\n-----------------\r\n";
            for (size_t tensorElementIndex = 0; tensorElementIndex < N; ++tensorElementIndex, ++outputBufferData)
                outputString += std::to_wstring(*outputBufferData) + L' ';
            outputString += L"\r\n\r\n";
        }

        o += Method3TensorSizes[2];
        outputBufferData = (float*)o;
        Sxy = outputBufferData[N - 1];
        if (N <= 5)
        {
            outputString += L"Output 4 - Sxy\r\n-----------------\r\n";
            for (size_t tensorElementIndex = 0; tensorElementIndex < N; ++tensorElementIndex, ++outputBufferData)
                outputString += std::to_wstring(*outputBufferData) + L' ';
            outputString += L"\r\n\r\n";
        }

        o += Method3TensorSizes[3];
        outputBufferData = (float*)o;
        if (N <= 5)
        {
            outputString += L"Output 5 - xx\r\n-----------------\r\n";
            for (size_t tensorElementIndex = 0; tensorElementIndex < N; ++tensorElementIndex, ++outputBufferData)
                outputString += std::to_wstring(*outputBufferData) + L' ';
            outputString += L"\r\n\r\n";
        }

        o += Method3TensorSizes[4];
        outputBufferData = (float*)o;
        Sx2 = outputBufferData[N - 1];
        if (N <= 5)
        {
            outputString += L"Output 6 - Sxx\r\n-----------------\r\n";
            for (size_t tensorElementIndex = 0; tensorElementIndex < N; ++tensorElementIndex, ++outputBufferData)
                outputString += std::to_wstring(*outputBufferData) + L' ';
            outputString += L"\r\n\r\n";
        }
    }
    else
    {
        for (size_t tensorElementIndex{ 0 }; tensorElementIndex < (tensorOutputSize / 4); ++tensorElementIndex, ++outputBufferData)
        {
            outputString += std::to_wstring(*outputBufferData) + L' ';
        }
    }
    if (N <= 5)
        std::wcout << L"Outputs\r\n-------------\r\n" << outputString << std::endl << std::endl;

    // B
    float B = (N * Sxy - Sx * Sy) / ((N * Sx2) - (Sx * Sx));
    float A = (Sy - (B * Sx)) / N;


    if (Method == 3)
        printf("Linear Regression GPU:\r\nSx = %f\r\nSy = %f\r\nSxy = %f\r\nSx2 = %f\r\nA = %f\r\nB = %f\r\n\r\n", Sx, Sy, Sxy, Sx2, A, B);

    D3D12_RANGE emptyRange{ 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);
    getchar();
}



