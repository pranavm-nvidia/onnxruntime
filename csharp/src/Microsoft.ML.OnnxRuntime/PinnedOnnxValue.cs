using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.OnnxRuntime
{
    /// <summary>
    /// A lightweighted class to wrap a numberic tensor OrtValue.
    /// When used as model outputs, the passed in shape have to match the actual output value.
    ///
    /// This class is a part of prototype and may be changed in future version.
    /// </summary>
    public class PinnedOnnxValue : IDisposable
    {
        internal MemoryHandle PinnedMemory { get; private set; }
        internal IntPtr Value { get; private set; }

        internal PinnedOnnxValue(MemoryHandle pinnedMemory, ulong sizeInBytes, long[] shape, TensorElementType elementType)
        {
            PinnedMemory = pinnedMemory;
            IntPtr status;
            unsafe
            {
                status = NativeMethods.OrtCreateTensorWithDataAsOrtValue(
                    NativeMemoryInfo.DefaultInstance.Handle,
                    (IntPtr)pinnedMemory.Pointer,
                    (UIntPtr)sizeInBytes,
                    shape,
                    (UIntPtr)shape.Length,
                    elementType,
                    out IntPtr Value);
            }
            NativeApiStatus.VerifySuccess(status);
        }

        public static PinnedOnnxValue Create(DenseTensor<int> data)
        {
            return Create(data.Buffer.Pin(), data.Buffer.Length * sizeof(int), data.Dimensions, TensorElementType.Int32);
        }
        public static PinnedOnnxValue Create(Memory<float> data, long[] shape)
        {
            return Create(data.Pin(), data.Length * sizeof(float), shape, TensorElementType.Float);
        }
        public static PinnedOnnxValue Create(Memory<long> data, long[] shape)
        {
            return Create(data.Pin(), data.Length * sizeof(long), shape, TensorElementType.Int64);
        }

        // TODO: support other element types

        // TODO: deal with unsupported OrtValue types, including string tensor, sequences and maps.

        private static PinnedOnnxValue Create(MemoryHandle pinnedMemory, int length, ReadOnlySpan<int> shape, TensorElementType elementType)
        {
            try
            {
                long[] shapeArray = new long[shape.Length];
                for (int i = 0; i < shape.Length; i++)
                {
                    shapeArray[i] = shape[i];
                }
                return new PinnedOnnxValue(pinnedMemory, (ulong)length, shapeArray, elementType);
            }
            catch
            {
                pinnedMemory.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
            ((IDisposable)PinnedMemory).Dispose();

            if (Value != IntPtr.Zero)
            {
                NativeMethods.OrtReleaseValue(Value);
            }
        }
    }
}
