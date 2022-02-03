// Copyright (c) 2015 ~ 2019 NVIDIA Corporation 
// 
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
// 
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

/// \file 
/// Header of the low-level interface of NVTT.
///
/// This contains functions for compressing to each of the formats NVTT
/// supports, as well as different ways of specifying the input and output.
/// For instance, the low-level API allows both compression input and output
/// to exist on the GPU, removing the need for any CPU-to-GPU or GPU-to-CPU
/// copies.
/// 
/// Here are some general notes on the low-level compression functions.
/// 
/// These functions often support "fast-mode" and "slow-mode" compression.
/// These switch between two compression algorithms: fast-mode algorithms
/// are faster but lower-quality, while slow-mode algorithms are slower
/// but higher-quality. Other functions support multiple quality levels.
/// 
/// Sometimes, the fast-mode or slow-mode algorithm isn't available on the GPU.
/// In that case, that compression will be done on the CPU instead. In this
/// case, slow-mode compression on the GPU may be faster than fast-mode
/// compression on the CPU.
/// 
/// To use GPU compression, there must be a GPU that supports CUDA.
/// See nvtt::isCudaSupported().

#ifndef nvtt_lowlevel_h
#define nvtt_lowlevel_h

#ifdef _DOXYGEN_
/// @brief Functions with this macro are accessible via the NVTT DLL.
#define NVTT_API
#endif

// Function linkage
#if NVTT_SHARED

#if defined _WIN32 || defined WIN32 || defined __NT__ || defined __WIN32__ || defined __MINGW32__
#  ifdef NVTT_EXPORTS
#    define NVTT_API __declspec(dllexport)
#  else
#    define NVTT_API __declspec(dllimport)
#  endif
#endif

#if defined __GNUC__ >= 4
#  ifdef NVTT_EXPORTS
#    define NVTT_API __attribute__((visibility("default")))
#  endif
#endif

#endif // NVTT_SHARED

#if !defined NVTT_API
#  define NVTT_API
#endif

namespace nvtt
{
	/// Check if CUDA is supported by the run-time environment
	NVTT_API bool isCudaSupported();

	/// @brief Tells NVTT to always use an application-set device rather
	/// than selecting its own.
	/// 
	/// By default, NVTT functions such as nvtt::isCudaSupported() and
	/// nvtt::Context() can choose a device and call cudaSetDevice().
	/// Calling this function will prevent NVTT from calling cudaSetDevice(),
	/// and will make it use the currently set device instead.
	/// The application must then call cudaSetDevice() before calling NVTT
	/// functions, and whenever it wants to change the device subsequent
	/// NVTT functions will use.
	/// 
	/// For instance, this may be useful when managing devices on multi-GPU
	/// systems.
	NVTT_API void useCurrentDevice();

	struct TimingContext;

	/// Value type of the input images.
	/// The input buffer will use the same value type as the input images
	enum ValueType
	{
		UINT8,
		SINT8,
		FLOAT32
	};

	/// Name of channels for defining a swizzling
	enum ChannelOrder
	{
		Red = 0,
		Green = 1,
		Blue = 2,
		Alpha = 3,
		Zero = 4,
		One = 5
	};

	/// Use this structure to reference each of the input images
	struct RefImage
	{
		const void *data; ///< For CPUInputBuffer, this should point to host memory; for GPUInputBuffer, this should point to device memory.
		int width = 0; ///< Width of the image. This can be arbitrary, up to 65535.
		int height = 0; ///< Height of the image. This can be arbitrary, up to 65535.
		int depth = 1; ///< Z-dimension of the images, usually 1
		int num_channels = 4; ///< Number of channels the image has
		ChannelOrder channel_swizzle[4] = { Red,Green,Blue,Alpha }; ///< Channels order how the image is stored
		bool channel_interleave = true; ///< Whether the rgba channels are interleaved (r0, g0, b0, a0, r1, g1, b1, a1, ...)
	};

	/// @brief Structure containing all the input images from host memory.
	/// The image data is reordered by tiles.
	struct CPUInputBuffer
	{
		/// @brief Construct a CPUInputBuffer from 1 or more RefImage structs.
		/// 
		/// The input images should use the same value type.
		/// `images[i].data` should point to host memory here.
		/// `num_tiles` can be an array of numImages elements used to return the number of tiles of each input image after reordering.
		NVTT_API CPUInputBuffer(const RefImage* images, ValueType value_type, int numImages=1, int tile_w=4, int tile_h=4,
			float WeightR=1.0f, float WeightG=1.0f, float WeightB=1.0f, float WeightA=1.0f, nvtt::TimingContext *tc = nullptr, unsigned* num_tiles = nullptr);
		
		/// Destructor
		NVTT_API ~CPUInputBuffer();

		/// The total number of tiles of the input buffer
		NVTT_API int NumTiles() const;

		/// Tile Size
		NVTT_API void TileSize(int& tile_w, int& tile_h) const;

		/// Value type. The same as the input images used for creating this object
		NVTT_API ValueType Type() const;

		/// Get a pointer to the i-th tile. Mainly used internally.
		NVTT_API void* GetTile(int i, int& vw, int& vh) const;

		struct Private;
		Private *m;
	};

	/// @brief Structure containing all the input images from device memory.
	/// The image data is reordered by tiles.
	struct GPUInputBuffer
	{
		/// @brief Construct a GPUInputBuffer from 1 or more RefImage structs.
		/// 
		/// The input images should use the same value type.
		/// `images[i].data` should point to GPU global memory here (more specifically, a CUDA device pointer).
		/// `num_tiles` can be used to return the number of tiles of each input image after reordering.
		NVTT_API GPUInputBuffer(const RefImage* images, ValueType value_type, int numImages = 1, int tile_w = 4, int tile_h = 4,
			float WeightR = 1.0f, float WeightG = 1.0f, float WeightB = 1.0f, float WeightA = 1.0f, nvtt::TimingContext *tc = nullptr, unsigned* num_tiles = nullptr);

		/// @brief Construct a GPUInputBuffer from a CPUInputBuffer.
		/// 
		/// `begin`/`end` can be used to sepcify a range of tiles to copy from the CPUInputBuffer.
		/// `end = -1` means the end of the input buffer.
		NVTT_API GPUInputBuffer(const CPUInputBuffer& cpu_buf, int begin = 0, int end = -1, nvtt::TimingContext *tc = nullptr);
		
		/// Destructor
		NVTT_API ~GPUInputBuffer();

		/// The total number of tiles of the input buffer
		NVTT_API int NumTiles() const;

		/// Tile Size
		NVTT_API void TileSize(int& tile_w, int& tile_h) const;

		/// @brief Value type. The same as the input images of the CPUInputBuffer used for creating this object
		NVTT_API ValueType Type() const;

		struct Private;
		Private *m;

	};

	///////////////////////////// BC1 ///////////////////////////////

	/// Interface for compressing to BC1 format from CPUInputBuffer.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param fast_mode If true, uses a faster but lower-quality compressor;
	/// otherwise, uses a slower but higher-quality compressor. This applies
	/// to both CPU and GPU compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc1(const CPUInputBuffer& input, bool fast_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC1 format from GPUInputBuffer, always
	/// using GPU compression.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param fast_mode If true, uses a faster but lower-quality compressor;
	/// otherwise, uses a slower but higher-quality compressor. Compression
	/// always happens on the GPU, so CUDA must be available.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc1(const GPUInputBuffer& input, bool fast_mode, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC1a ///////////////////////////////

	/// Interface for compressing to BC1a format from CPUInputBuffer.
	/// No fast-mode algorithm for the GPU is available, so when `fast_mode`
	/// is true this ignores `useGPU` and compresses on the CPU. In this case,
	/// slow-mode GPU compression may be faster than fast-mode CPU compression.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param fast_mode If true, uses a faster but lower-quality compressor;
	/// otherwise, uses a slower but higher-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for fast-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `fast_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc1a(const CPUInputBuffer& input, bool fast_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC1a format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU slow-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc1a(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC2 ///////////////////////////////
	
	/// Interface for compressing to BC2 format from CPUInputBuffer.
	/// No fast-mode algorithm for the GPU is available, so when `fast_mode`
	/// is true this ignores `useGPU` and compresses on the CPU. In this case,
	/// slow-mode GPU compression may be faster than fast-mode CPU compression.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param fast_mode If true, uses a faster but lower-quality compressor;
	/// otherwise, uses a slower but higher-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for fast-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `fast_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc2(const CPUInputBuffer& input, bool fast_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC2 format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU slow-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc2(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC3 ///////////////////////////////

	/// Interface for compressing to BC3 format from CPUInputBuffer.
	/// No fast-mode algorithm for the GPU is available, so when `fast_mode`
	/// is true this ignores `useGPU` and compresses on the CPU. In this case,
	/// slow-mode GPU compression may be faster than fast-mode CPU compression.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param fast_mode If true, uses a faster but lower-quality compressor;
	/// otherwise, uses a slower but higher-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for fast-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `fast_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc3(const CPUInputBuffer& input, bool fast_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC3 format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU slow-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc3(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC3n ///////////////////////////////

	/// Interface for compressing to BC3n format from CPUInputBuffer.
	/// This method is currently CPU-only, but supports 3 quality levels
	/// - 0, 1, and 2.
	/// See nvtt::Format_BC3n.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param qualityLevel Higher quality levels produce less compression
	/// error, but take longer to compress. Can be 0, 1, or 2.
	/// 
	/// @param output Pointer to output in CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc3n(const CPUInputBuffer& input, int qualityLevel, void* output, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC3 - rgbm ///////////////////////////////

	/// Interface for compressing to BC3 - rgbm format from CPUInputBuffer.
	/// This method is currently CPU-only and has 1 quality level.
	/// See nvtt::Format_BC3_RGBM.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param output Pointer to output in CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc3_rgbm(const CPUInputBuffer& input, void* output, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC4U ///////////////////////////////

	/// Interface for compressing to BC4U format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc4(const CPUInputBuffer& input, bool slow_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC4U format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc4(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC4S ///////////////////////////////

	/// Interface for compressing to BC4S format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc4s(const CPUInputBuffer& input, bool slow_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC4S format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc4s(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// ATI2 ///////////////////////////////

	/// Interface for compressing to ATI2 format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_ati2(const CPUInputBuffer& input, bool slow_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to ATI2 format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_ati2(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC5U ///////////////////////////////

	/// Interface for compressing to BC5U format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc5(const CPUInputBuffer& input, bool slow_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC5U format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc5(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC5S ///////////////////////////////

	/// Interface for compressing to BC5S format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc5s(const CPUInputBuffer& input, bool slow_mode, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC5S format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc5s(const GPUInputBuffer& input, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC7 ///////////////////////////////

	/// Interface for compressing to BC7 format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU. The slow-mode
	/// CPU compressor is particularly slow in this case (as it searches
	/// though a very large space of possibilities), so fast-mode compression
	/// is recommended.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param imageHasAlpha Specifies that some pixels in the image have an
	/// alpha value less than 1.0f. If false, this makes compression slightly
	/// faster. It's still valid to set it to true even if the image is opaque.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc7(const CPUInputBuffer& input, bool slow_mode, bool imageHasAlpha, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC7 format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param imageHasAlpha Specifies that some pixels in the image have an
	/// alpha value less than 1.0f. If false, this makes compression slightly
	/// faster. It's still valid to set it to true even if the image is opaque.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc7(const GPUInputBuffer& input, bool imageHasAlpha, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// BC6H ///////////////////////////////

	/// Interface for compressing to BC6H format from CPUInputBuffer.
	/// No slow-mode algorithm for the GPU is available, so when `slow_mode`
	/// is true this ignores `useGPU` and compresses on the CPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param slow_mode If true, uses a slower but higher-quality compressor;
	/// otherwise, uses a faster but lower-quality compressor. Also overrides
	/// `useGPU` if true and uses the CPU for slow-mode compression.
	/// 
	/// @param is_signed If true, compresses to the BC6S format, instead of BC6U.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc6h(const CPUInputBuffer& input, bool slow_mode, bool is_signed, void* output, bool useGpu = false, bool to_device_mem = false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to BC6H format from GPUInputBuffer, always
	/// using GPU compression. This method has only one quality level,
	/// corresponding to CPU fast-mode.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param is_signed If true, compresses to the BC6S format, instead of BC6U.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_bc6h(const GPUInputBuffer& input, bool is_signed, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////

	///////////////////////////// ASTC ///////////////////////////////

	/// Interface for compressing to ASTC format from CPUInputBuffer.
	/// This supports 4 quality levels on both the CPU and GPU.
	/// 
	/// @param input Constant reference to input in CPU memory.
	/// 
	/// @param qualityLevel The quality level, 0, 1, 2, or 3. Higher quality
	/// levels produce less compression error, but take longer.
	/// 
	/// @param imageHasAlpha Specifies that some pixels in the image have an
	/// alpha value less than 1.0f. If false, this makes compression slightly
	/// faster. It's still valid to set it to true even if the image is opaque.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param useGpu Whether to run the compression algorithm on the GPU as
	/// opposed to the CPU. See note on `slow_mode`.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_astc(const CPUInputBuffer& input, int qualityLevel, bool imageHasAlpha, void* output, bool useGpu = false, bool to_device_mem= false, nvtt::TimingContext *tc = nullptr);

	/// Interface for compressing to ASTC format from GPUInputBuffer, always
	/// using GPU compression. This supports 4 quality levels.
	/// 
	/// @param input Constant reference to input in GPU memory.
	/// 
	/// @param qualityLevel The quality level, 0, 1, 2, or 3. Higher quality
	/// levels produce less compression error, but take longer.
	/// 
	/// @param imageHasAlpha Specifies that some pixels in the image have an
	/// alpha value less than 1.0f. If false, this makes compression slightly
	/// faster. It's still valid to set it to true even if the image is opaque.
	/// 
	/// @param output Pointer to output: CUDA device pointer if `to_device_mem`
	/// is true, and a pointer to CPU memory otherwise.
	/// 
	/// @param to_device_mem Specifies that `output` is a CUDA device pointer,
	/// rather than a pointer to CPU memory.
	/// 
	/// @param tc Timing context for recording performance information.
	void NVTT_API nvtt_encode_astc(const GPUInputBuffer& input, int qualityLevel, bool imageHasAlpha, void* output, bool to_device_mem = true, nvtt::TimingContext *tc = nullptr);

	//////////////////////////////////////////////////////////////////
}

#endif // nvtt_lowlevel_h
