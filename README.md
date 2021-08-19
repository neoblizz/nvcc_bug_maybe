# NVCC Bug? 
The goal of this code is to showcase a possible compiler bug with NVCC. The bug seems to happen when a kernel uses a lambda function that captures an object by value. The code is too simple to parse (less than 50 lines). The relevant part shown below where `Vector` is a simple class that defines its own copy constructor, destructor, and assignment operator 

```
template <typename T>
__global__ void mainKernel(Vector<T> vec)
{    
    auto print_vector_size = [=]() {
        printf("\n vec size = %d\n", vec.m_size);
    };

    print_vector_size();    
}
```
Changing the lambda function to capture by reference ([&]) fixes the bug. 


## Build 
```
git clone https://github.com/Ahdhn/nvcc_bug_maybe.git
cd nvcc_bug_maybe
mkdir build
cd build 
cmake ..
```
Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 

## Output
On GCC 9.3.0 + CUDA 11.2, we get the following during the compilation 
```
Segmentation fault (core dumped)
```
or check [here](https://github.com/Ahdhn/nvcc_bug_maybe/runs/3367653261?check_suite_focus=true#step:7:20) for On GCC 9.3.0 + CUDA 11.3

On Visual Studio 2019 + CUDA 11.4, at one step during the compilation, we get the following from the Output windows 
```
Operand is null
call void @_ZN6VectorIiEC1ERKS0_(%struct._Z6VectorIiE* %tmp, <null operand!>), !dbg !9
Wrote crash dump file "C:\Users\engah\AppData\Local\Temp\cicc.exe-9d437b.dmp"
LLVMSymbolizer: error reading file: 'kernel32.pdb': no such file or directory
LLVMSymbolizer: error reading file: 'ntdll.pdb': no such file or directory
#0 0x00007ff6672aa358 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0x56a358 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0x5408c8
#1 0x00007ff6672aa358 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0x542849 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0x38384d
#2 0x00007ff6672aa358 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0x37b876 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0x387bff
#3 0x00007ff6672aa358 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe 0xf23f34 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe+0x56a358)
#4 0x00007ff6672aa358
#5 0x00007ff6672aa358 (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\nvvm\bin\cicc.exe+0x56a358)
0x00007FF6672AA358 (0x0000029B4D792158 0x0000000000000006 0x0000029B4CE62530 0x00007FF66728088C)
0x00007FF6672808C8 (0x0000000000000000 0x0000029B4D10E7A0 0x000000A74E1FE9E0 0x00007FF6673F0000)
0x00007FF667282849 (0x00007FF6686487D0 0x0000000000000248 0x0000000000000248 0x00007FFE4492A305)
0x00007FF6670C384D (0x0000029B4AFD4B60 0x00007FF668657568 0x0000029B4AFD4B60 0x0000029B4AFD4B60)
0x00007FF6670BB876 (0x00007FF668657568 0x0000000000000020 0x0000000000000020 0x00007FF668657568)
0x00007FF6670C7BFF (0x00007FF668657568 0x0000000000000000 0x0000029B4AFEB080 0x0000000000000000)
0x00007FF667C63F34 (0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000)
0x00007FFE45B47034 (0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000), BaseThreadInitThunk() + 0x14 bytes(s)
0x00007FFE46BC2651 (0x0000000000000000 0x0000000000000000 0x0000000000000000 0x0000000000000000), RtlUserThreadStart() + 0x21 bytes(s)
main.cu
CUDACOMPILE : nvcc error : 'cicc' died with status 0xC0000005 (ACCESS_VIOLATION)
```
or check [here](https://github.com/Ahdhn/nvcc_bug_maybe/runs/3367653262?check_suite_focus=true#step:7:24) for On Visual Studio 2019 + CUDA 11.3
