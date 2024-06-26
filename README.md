# DISCONTINUATION OF PROJECT #  
This project will no longer be maintained by Intel.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  
 If you have an ongoing need to use this project, are interested in independently developing it, or would like to maintain patches for the open source software community, please create your own fork of this project.  
  
# CmdThrottlePolicy

This sample demonstrates the use of the Command Throttle Policy extension with a simple matrix multiplication compute shader kernel.

To validate the extension is working correctly, just build and run the sample. If you want to verify the performance difference with
the extension engaged vs. not engaged, run it one time without any parameters, and run it again with the additional flag
`--disable-command-throttle-policy-extension`.

The sample relies on an external dependency, `DXSampleHelper.h` from the DirectX SDK Samples available at https://github.com/microsoft/DirectX-Graphics-Samples/tree/master. Copy this to the project folder.

This sample relies on the Intel Extensions static library available at: https://github.com/GameTechDev/64-bit-Typed-Atomics-Extension/blob/main/INTC_Atomic_64bit_Max/bin/x64/Debug/igdext64.lib or https://github.com/GameTechDev/D3DExtensions_public. Copy this to `..\CmdThrottlePolicy\IntelExtension\lib`. 

Supported command line parameters:
- --disable-command-throttle-policy-extension\
  Don't use the command throttle policy extension. By default we will set the command throttle\
  policy to MAX_PERFORMANCE with the Command Throttle Policy Extension.

- --check-gpu-result\
  Do matrix multiplication on CPU and compare the result with the one on GPU.

- -h\
  Print helper information.
