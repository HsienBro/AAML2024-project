# AAML 2024 Final Project

## 專案介紹 (Project Overview)
本專案為 NYCU-AAML 2024 的期末專案。
基於 [CFU-Playground](https://github.com/google/CFU-Playground) 框架，透過Verilog實作Systolic Array在FPGA上加速CNN模型的卷積層(Convolution Layer)中的矩陣乘法運算。

## Features
- **硬體加速器 (Hardware Accelerator)**: 實作了4*4 Systolic Array來加速模型中的矩陣乘法運算。
- **軟體層優化 (Software Optimization)**: 在軟體層面利用Loop Unrolling，降低迴圈迭代時的開銷，讓CPU更有效率的進行運算。

## 原始碼位置 (Source Code Location)
- `proj\AAML_final_proj\cfu.v`: 這是一個實作了4*4 Systolic Array的Verilog程式碼。
- `proj\AAML_final_proj\src\tensorflow\lite\kernels\internal\reference\integer_ops\conv.h`: 這是在軟體層的Conv Layer中，利用Img2Col和Loop Unrolling進行優化的C++程式碼。

## 效能表現 (Performance / Results)
- **Baseline**: `4044074` μs 
- **Accelerated**: `424153` μs
- **Speedup**: `9.53` 倍加速


## Original CFU Playground README
# CFU Playground

Want a faster ML processor?   Do it yourself!

This project provides a framework that an engineer, intern, or student can use to design and evaluate **enhancements** to an FPGA-based “soft” processor, specifically to increase the performance of machine learning (ML) tasks.   The goal is to abstract away most infrastructure details so that the user can get up to speed quickly and focus solely on adding new processor instructions, exploiting them in the computation, and measuring the results.

This project enables rapid iteration on processor improvements -- multiple iterations per day.

This is how it works:
* Choose a TensorFlow Lite model; a quantized person detection model is provided, or bring your own.
* Execute the inference on the Arty FPGA board to get cycle counts per layer.
* Choose an TFLite operator to accelerate, and dig into that code.
* Design new instruction(s) that can replace multiple basic operations.
* Build a custom function unit (a small amount of hardware) that performs the new instruction(s).
* Modify the TFLite/Micro library kernel to use the new instruction(s), which are available as intrinsics with function call syntax.
* Rebuild the FPGA Soc, recompile the TFLM library, and rerun to measure improvement.

The focus here is performance, not demos.  The inputs to the ML inference are canned/faked, and the only output is cycle counts.  It would be possible to export the improvements made here to an actual demo, but currently no pathway is set up for doing so.

With the exception of Vivado, everything used by this project is open source.

**Disclaimer: This is not an officially supported Google project.   Support and/or new releases may be limited.**

_This is an early prototype of a ML exploration framework; expect a lack of documentation and occasional breakage. If you want to collaborate on building out this framework, reach out to tcal@google.com!   See "Contribution guidelines" below._


### Required hardware/OS

* One of the boards supported by [LiteX Boards](https://github.com/litex-hub/litex-boards/tree/master/litex_boards/targets). Most of LiteX Boards targets should work.\
It has been tested on the **Arty A7-35T/100T**, **iCEBreaker**, **Fomu**, **OrangeCrab**, **ULX3S**, and **Nexys Video** boards.
* The only supported host OS is Linux (Debian / Ubuntu).

You don't need any board if you want to run [Renode](https://renode.io) or Verilator simulation.

### Assumed software

* FPGA Toolchain: that depends on a chosen board.  If you already have a toolchain installed for your board, you can use that.

For a board with a Xilinx XC7 part, you can use either [Vivado](https://www.xilinx.com/support/download.html),
which must be manually installed (here's our [guide](https://cfu-playground.readthedocs.io/en/latest/vivado-install.html)),
or the open-source SymbiFlow tool chain, which can be easily installed using Conda
(see the [Setup Guide](https://cfu-playground.readthedocs.io/en/latest/setup-guide.html)).

For boards with Lattice iCE40, ECP5, or Nexus FPGAs, you can install the appropriate set of open source tools
either via Conda (see the [Setup Guide](https://cfu-playground.readthedocs.io/en/latest/setup-guide.html))
or on your own by building from source.   Or, you can use the Lattice toolchain (Radiant/Diamond).

If you want to try things out using [Renode](https://renode.io) simulation, then you don't need either the board or toolchain.
You can also perform Verilog-level cycle-accurate simulation with Verilator, but this is much slower.
Renode is installed by the setup script.

Other required packages will be checked for and, if on a Debian-based system, automatically installed by the setup script below.


### Setup

Clone this repo, `cd` into it, then get run:
```sh
scripts/setup
```

### Use with board

The default board is Arty. If you want to use different board you must specify target, e.g. `TARGET=digilent_nexys_video`.
1. Build the SoC and load the bitstream onto Arty:
```sh
cd proj/proj_template
make prog
```

This builds the SoC with the default CFU from `proj/proj_template`. Later you'll copy this and modify it to make your own project.


2. Build a RISC-V program and execute it on the SoC that you just loaded onto the Arty:
```sh
make load
```

### Use without board

If you don't have any board supported by LiteX Boards you can use Renode or Verilator to simulate it.

To use Renode to execute on a simulator on the host machine (no Vivado or Arty board required), execute:

```sh
make renode
```

To use Verilator to execute on a cycle-accurate RTL-level simulator (no Vivado or Arty board required), execute:

```sh
make PLATFORM=sim load
```

### Most useful make flags

| Option          | Explanation   | Example | Default |
| --------------- | ------------- | ------- | ------- |
| `PLATFORM`      | Choose which SoC platform you want to build: `hps` or `sim` or `common_soc` | `make bitstream PLATFORM=hps` | `common_soc` |
| `TARGET`        | Choose one of many targets from [LiteX Boards](https://github.com/litex-hub/litex-boards/tree/master/litex_boards/targets) repository, `common_soc` will take `BaseSoC` from specified `target.py` | `make bitstream TARGET=nexys_video_board` | `digilent_arty` |
| `USE_VIVADO`    | Use Vivado toolchain | `make bitstream USE_VIVADO=1` | `0` |
| `USE_SYMBIFLOW` | Use Symbiflow toolchain | `make bitstream USE_SYMBIFLOW=1` | `0` |
| `UART_SPEED`    | Choose UART baudrate | `make bitstream UART_SPEED=115200` | `3686400` |
| `IGNORE_TIMING` | Ignore timing contraints (only for Vivado) | `make bitstream USE_VIVADO=1 IGNORE_TIMING=1` | `0` |

### Underlying open-source technology

* [LiteX](https://github.com/enjoy-digital/litex): Open-source framework for assembling the SoC (CPU + peripherals)
* [VexRiscv](https://github.com/SpinalHDL/VexRiscv): Open-source RISC-V soft CPU optimized for FPGAs
* [Amaranth](https://github.com/amaranth-lang/amaranth): Python toolbox for building digital hardware


### Licensed under Apache-2.0 license

See the file [LICENSE](LICENSE).

### Contribution guidelines

**If you want to contribute to CFU Playground, be sure to review the
[contribution guidelines](CONTRIBUTING.md).  This project adheres to Google's
[code of conduct](CODE_OF_CONDUCT.md).   By participating, you are expected to
uphold this code.**
