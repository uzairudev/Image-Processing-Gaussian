# Image Processing - Gaussian Filter

This repository contains a C++ program that applies a Gaussian filter to an image using OpenCV. Gaussian filtering is widely used for reducing image noise and detail by applying a low-pass filter.

## Features

- Customizable kernel size and sigma values for Gaussian filter.
- Iterative performance benchmarking with output for total time, time per iteration, and iterations per second (IPS).
- Supports multithreading using OpenMP for faster performance.

## Prerequisites

- [OpenCV 4.x](https://opencv.org/) or higher.
- C++ compiler with OpenMP support.

## Compilation Instructions

Use the following commands to compile the program:

```bash
g++ gaussian_filter.cpp -fopenmp `pkg-config opencv4 --cflags` -c
g++ gaussian_filter.o -fopenmp `pkg-config opencv4 --libs` -lstdc++ -o gaussian_filter
```

## How to Run

After compiling, you can run the program:

```bash
./gaussian_filter
```

During execution, the program will prompt you to enter the kernel size (must be an odd integer) and the sigma value.

Example:

```bash
Enter kernel size (must be an odd integer): 5
Enter sigma value: 1.5
```

## Gaussian Filter

The Gaussian filter is computed using the following formula for each kernel value:

\[
G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
\]

- `Kernel size`: Specifies the dimensions of the Gaussian filter (e.g., 3x3, 5x5).
- `Sigma`: Controls the spread of the Gaussian function. Larger sigma values lead to more blurring.

## Example

Make sure to have an image file named `stereo_image.jpg` in the working directory. The program will load this image, apply the Gaussian filter, and display the original and filtered images.

## Performance

The program benchmarks the filtering process over multiple iterations (default: 100 iterations) and provides the following metrics:

- Total execution time.
- Time per iteration.
- Iterations per second (IPS).

