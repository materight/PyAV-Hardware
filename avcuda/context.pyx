cimport libav
from libc.stdint cimport uint8_t
from av.video.frame cimport VideoFrame
from av.codec.context cimport CodecContext
import torch

from libc.stdio cimport printf

from avcuda cimport libavhw, cuda
from avcuda.libavhw cimport AVBufferRef, AVHWDeviceType, AVCodecContext, AVHWFramesContext


cdef class HWDeviceContext:

    cdef AVBufferRef* ptr
    cdef int device

    def __cinit__(self, int device):
        self.ptr = NULL
        self.device = device

        cdef err = libavhw.av_hwdevice_ctx_create(
            &self.ptr,
            libavhw.AV_HWDEVICE_TYPE_CUDA,
            str(self.device).encode(),
            NULL,
            0
        )
        if err < 0:
            raise RuntimeError(f"Failed to create specified HW device. {libav.av_err2str(err).decode('utf-8')}.")

    def close(self):
        if self.ptr:
            libavhw.av_buffer_unref(&self.ptr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def attach_decoder(self, CodecContext codec_context):
        cdef AVCodecContext* ctx = <AVCodecContext*> codec_context.ptr
        ctx.hw_device_ctx = libavhw.av_buffer_ref(self.ptr)

    def attach_encoder(self, CodecContext codec_context):
        cdef AVCodecContext* ctx = <AVCodecContext*> codec_context.ptr
        ctx.hw_device_ctx = libavhw.av_buffer_ref(self.ptr)
        ctx.sw_pix_fmt = ctx.pix_fmt
        ctx.pix_fmt = libavhw.AV_PIX_FMT_CUDA

        ctx.hw_frames_ctx = libavhw.av_hwframe_ctx_alloc(self.ptr)
        if not ctx.hw_frames_ctx:
            raise RuntimeError("Failed to allocate CUDA frame context.")

        cdef AVHWFramesContext* frames_ctx = <AVHWFramesContext*> ctx.hw_frames_ctx.data
        frames_ctx.format = ctx.pix_fmt
        frames_ctx.sw_format = ctx.sw_pix_fmt
        frames_ctx.width = ctx.width
        frames_ctx.height = ctx.height
        frames_ctx.initial_pool_size = 5

        cdef err = libavhw.av_hwframe_ctx_init(ctx.hw_frames_ctx)
        if err < 0:
            raise RuntimeError(f"Failed to initialize CUDA frame context. {libav.av_err2str(err).decode('utf-8')}.")

    def to_tensor(self, frame: VideoFrame) -> torch.Tensor:
        tensor = torch.empty((frame.ptr.height, frame.ptr.width, 3), dtype=torch.uint8, device=torch.device('cuda', self.device))
        cdef cuda.CUdeviceptr tensor_ptr = tensor.data_ptr()
        with nogil:
            err = cuda.NV12ToRGB(
                <uint8_t*> frame.ptr.data[0],
                <uint8_t*> frame.ptr.data[1],
                <uint8_t*> tensor_ptr,
                frame.ptr.height,
                frame.ptr.width,
                frame.ptr.linesize[0],
                (frame.ptr.color_range == libav.AVCOL_RANGE_JPEG), # Use full color range for yuvj420p format
            )
            if err != cuda.cudaSuccess:
                raise RuntimeError(f"Failed to decode CUDA frame: {cuda.cudaGetErrorString(err).decode('utf-8')}.")
        return tensor

    def from_tensor(self, CodecContext codec_context, tensor: torch.Tensor) -> VideoFrame:
        cdef cuda.CUdeviceptr tensor_ptr = tensor.data_ptr()
        cdef int height = tensor.shape[0]
        cdef int width = tensor.shape[1]
        frame = VideoFrame(0, 0, format="cuda") # Allocate an empty frame with the final format
        with nogil:
            frame.ptr = libav.av_frame_alloc()
            frame.ptr.height = height
            frame.ptr.width = width
            frame.ptr.pts = 0
            frame.ptr.format = libavhw.AV_PIX_FMT_CUDA
            err = libavhw.av_hwframe_get_buffer((<AVCodecContext*> codec_context.ptr).hw_frames_ctx, frame.ptr, 0)
            if err < 0:
                raise RuntimeError(f"Failed to allocate CUDA frame: {libav.av_err2str(err).decode('utf-8')}.")
            printf("%d\n", frame.ptr.height)
            printf("%d\n", frame.ptr.width)
            #printf("%d\n", frame.ptr.linesize[0])

            err_cuda = cuda.RGBToNV12(
                <uint8_t*> tensor_ptr,
                <uint8_t*> frame.ptr.data[0],
                <uint8_t*> frame.ptr.data[1],
                frame.ptr.height,
                frame.ptr.width,
                frame.ptr.width,
            )
            if err != cuda.cudaSuccess:
                raise RuntimeError(f"Failed to encode CUDA frame: {cuda.cudaGetErrorString(err_cuda).decode('utf-8')}.")
        return frame
