﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using BrightData.Cuda.Helper;
using BrightData.Helper;
using BrightData.LinearAlegbra2;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using Microsoft.Toolkit.HighPerformance.Buffers;

namespace BrightData.Cuda
{
    public class CudaComputationUnit : ComputationUnit
    {
        readonly CudaProvider _cuda;

        public CudaComputationUnit(BrightDataContext context, CudaProvider cuda) : base(context)
        {
            _cuda = cuda;
        }

        public override IDisposableTensorSegment CreateSegment(uint size) => new CudaTensorSegment(_cuda.Allocate(size));
        public override IVector CreateVector(ITensorSegment2 data) => new CudaVector2(OptionallyCopyToDevice(data), this);
        public override IMatrix CreateMatrix(ITensorSegment2 data, uint rowCount, uint columnCount) => new CudaMatrix2(OptionallyCopyToDevice(data), rowCount, columnCount, this);

        public override IVector CreateVector(uint size, Func<uint, float> initializer)
        {
            using var buffer = SpanOwner<float>.Allocate((int)size);
            var ptr = buffer.Span;
            for (uint i = 0; i < size; i++)
                ptr[(int)i] = initializer(i);
            var deviceMemory = _cuda.Memory.GetMemory(size);
            deviceMemory.CopyToDevice(ptr, buffer.DangerousGetArray().Array);
            return CreateVector(new CudaTensorSegment(deviceMemory));
        }

        public override IMatrix CreateMatrix(uint rowCount, uint columnCount, Func<uint, uint, float> initializer)
        {
            var size = rowCount * columnCount;
            using var buffer = SpanOwner<float>.Allocate((int)size);
            var ptr = buffer.Span;
            for (uint i = 0; i < columnCount; i++) {
                for (uint j = 0; j < rowCount; j++)
                    ptr[(int)(j * columnCount + i)] = initializer(j, i);
            }
            var deviceMemory = _cuda.Memory.GetMemory(size);
            deviceMemory.CopyToDevice(ptr, buffer.DangerousGetArray().Array);
            return CreateMatrix(new CudaTensorSegment(deviceMemory), rowCount, columnCount);
        }

        public override IDisposableTensorSegment Clone(ITensorSegment2 tensor)
        {
            var ret = (CudaTensorSegment)CreateSegment(tensor.Size);
            ret.DeviceMemory.CopyToDevice(GetDeviceMemoryPtr(tensor));
            return ret;
        }

        ITensorSegment2 OptionallyCopyToDevice(ITensorSegment2 segment)
        {
            if (segment.SegmentType == "cuda")
                return segment;

            var deviceMemory = _cuda.Memory.GetMemory(segment.Size);
            deviceMemory.CopyToDevice(segment.GetSpan(), segment.GetArrayForLocalUseOnly());
            return new CudaTensorSegment(deviceMemory);
        }

        public override float DotProduct(ITensorSegment2 tensor, ITensorSegment2 tensor2)
        {
            return _cuda.Blas.Dot(
                GetDeviceVariable(tensor), 
                1, 
                GetDeviceVariable(tensor2), 
                1
            );
        }

        public override ITensorSegment2 Add(ITensorSegment2 tensor, float scalar)
        {
            // TODO: create new cuda method?
            using var buffer = SpanOwner<float>.Allocate((int)tensor.Size);
            Array.Fill(buffer.DangerousGetArray().Array!, scalar);
            var ret = (CudaTensorSegment)CreateSegment(tensor.Size);
            ret.CopyFrom(buffer.Span, buffer.DangerousGetArray().Array);

            _cuda.AddInPlace(ret.DeviceMemory, GetDeviceMemoryPtr(tensor), tensor.Size, 1f, 1f);
            return ret;
        }

        public override ITensorSegment2 Add(ITensorSegment2 tensor, ITensorSegment2 tensor2)
        {
            var size = GetSize(tensor, tensor2);
            var ret = (CudaTensorSegment)CreateSegment(size);
            ret.DeviceMemory.CopyToDevice(GetDeviceMemoryPtr(tensor2));
            _cuda.Blas.Axpy(1.0f, GetDeviceVariable(tensor), 1, ret.DeviceMemory.DeviceVariable, 1);
            return ret;
        }

        public override void AddInPlace(ITensorSegment2 target, ITensorSegment2 other)
        {
            var size = GetSize(target, other);
            _cuda.AddInPlace(GetDeviceMemoryPtr(target), GetDeviceMemoryPtr(other), size, 1f, 1f);
        }

        public override void AddInPlace(ITensorSegment2 target, ITensorSegment2 other, float coefficient1, float coefficient2)
        {
            var size = GetSize(target, other);
            _cuda.AddInPlace(GetDeviceMemoryPtr(target), GetDeviceMemoryPtr(other), size, coefficient1, coefficient2);
        }

        public override ITensorSegment2 Add(ITensorSegment2 tensor, ITensorSegment2 tensor2, float coefficient1, float coefficient2)
        {
            var size = GetSize(tensor, tensor2);
            var ret = (CudaTensorSegment)CreateSegment(size);
            ret.DeviceMemory.CopyToDevice(GetDeviceMemoryPtr(tensor));
            _cuda.AddInPlace(ret.DeviceMemory, GetDeviceMemoryPtr(tensor2), size, coefficient1, coefficient2);
            return ret;
        }

        public override void AddInPlace(ITensorSegment2 target, float scalar)
        {
            // TODO: create new cuda method?
            using var buffer = SpanOwner<float>.Allocate((int)target.Size);
            Array.Fill(buffer.DangerousGetArray().Array!, scalar);
            var ret = (CudaTensorSegment)CreateSegment(target.Size);
            ret.CopyFrom(buffer.Span, buffer.DangerousGetArray().Array);

            _cuda.AddInPlace(GetDeviceMemoryPtr(target), ret.DeviceMemory, target.Size, 1f, 1f);
        }

        public override void ConstrainInPlace(ITensorSegment2 segment, float? minValue, float? maxValue)
        {
            var ptr = GetDeviceMemoryPtr(segment);
            _cuda.Constrain(ptr, segment.Size, minValue ?? float.MinValue, maxValue ?? float.MaxValue);
            base.ConstrainInPlace(segment, minValue, maxValue);
        }

        public override float CosineDistance(ITensorSegment2 tensor, ITensorSegment2 other)
        {
            var size = GetSize(tensor, other);
            return _cuda.CosineDistance(GetDeviceMemoryPtr(tensor), GetDeviceMemoryPtr(other), size);
        }

        public override ITensorSegment2 LeakyReluDerivative(ITensorSegment2 tensor)
        {
            var ret = _cuda.LeakyReluDerivative(GetDeviceMemoryPtr(tensor), tensor.Size);
            return new CudaTensorSegment(ret);
        }

        public override ITensorSegment2 PointwiseMultiply(ITensorSegment2 tensor1, ITensorSegment2 tensor2)
        {
            var size = GetSize(tensor1, tensor2);
            var ret = _cuda.PointwiseMultiply(GetDeviceMemoryPtr(tensor1), GetDeviceMemoryPtr(tensor1), size);
            return new CudaTensorSegment(ret);
        }

        public override void SubtractInPlace(ITensorSegment2 target, ITensorSegment2 other)
        {
            var size = GetSize(target, other);
            _cuda.SubtractInPlace(GetDeviceMemoryPtr(target), GetDeviceMemoryPtr(other), size, 1f, 1f);
            base.SubtractInPlace(target, other);
        }

        public override void SubtractInPlace(ITensorSegment2 target, ITensorSegment2 other, float coefficient1, float coefficient2)
        {
            var size = GetSize(target, other);
            _cuda.SubtractInPlace(GetDeviceMemoryPtr(target), GetDeviceMemoryPtr(other), size, coefficient1, coefficient2);
            base.SubtractInPlace(target, other);
        }

        public override ITensorSegment2 Multiply(ITensorSegment2 target, float scalar)
        {
            var ret = (CudaTensorSegment)CreateSegment(target.Size);
            ret.DeviceMemory.CopyToDevice(GetDeviceMemoryPtr(target));
            _cuda.Blas.Scale(scalar, ret.DeviceMemory.DeviceVariable, 1);
            return ret;
        }

        public override float L2Norm(ITensorSegment2 segment) => _cuda.Blas.Norm2(GetDeviceVariable(segment), 1);

        public override ITensorSegment2 Subtract(ITensorSegment2 tensor1, ITensorSegment2 tensor2)
        {
            var size = GetSize(tensor1, tensor2);
            var ret = (CudaTensorSegment)CreateSegment(size);
            ret.DeviceMemory.CopyToDevice(GetDeviceMemoryPtr(tensor1));
            _cuda.Blas.Axpy(-1.0f, GetDeviceVariable(tensor2), 1, ret.DeviceMemory.DeviceVariable, 1);
            return ret;
        }

        public override ITensorSegment2 PointwiseDivide(ITensorSegment2 tensor1, ITensorSegment2 tensor2)
        {
            var size = GetSize(tensor1, tensor2);
            var ret = _cuda.PointwiseDivide(GetDeviceMemoryPtr(tensor1), GetDeviceMemoryPtr(tensor2), size);
            return new CudaTensorSegment(ret);
        }

        public override ITensorSegment2 Subtract(ITensorSegment2 tensor1, ITensorSegment2 tensor2, float coefficient1, float coefficient2)
        {
            var size = GetSize(tensor1, tensor2);
            var ret = (CudaTensorSegment)CreateSegment(size);
            ret.DeviceMemory.CopyToDevice(GetDeviceMemoryPtr(tensor1));
            _cuda.SubtractInPlace(ret.DeviceMemory, GetDeviceMemoryPtr(tensor2), size, coefficient1, coefficient2);
            return ret;
        }

        public override float L1Norm(ITensorSegment2 segment)
        {
            var abs = Abs(segment);
            try {
                return _cuda.SumValues(GetDeviceMemoryPtr(abs), segment.Size);
            }
            finally {
                abs.Release();
            }
        }

        public override (float Min, float Max, uint MinIndex, uint MaxIndex) GetMinAndMaxValues(ITensorSegment2 segment)
        {
            var ptr = GetDeviceMemoryPtr(segment);
            var (min, max) = _cuda.FindMinAndMax(ptr, segment.Size);
            var maxIndex = (uint)_cuda.Blas.Max(ptr.DeviceVariable, 1) - 1;
            var minIndex = (uint)_cuda.Blas.Min(GetDeviceVariable(segment), 1) - 1;
            return (min, max, minIndex, maxIndex);
        }
        public override float GetMin(ITensorSegment2 segment) => _cuda.FindMinAndMax(GetDeviceMemoryPtr(segment), segment.Size).Min;
        public override float GetMax(ITensorSegment2 segment) => _cuda.FindMinAndMax(GetDeviceMemoryPtr(segment), segment.Size).Max;
        public override uint GetMinIndex(ITensorSegment2 segment) => (uint)_cuda.Blas.Min(GetDeviceVariable(segment), 1) - 1;
        public override uint GetMaxIndex(ITensorSegment2 segment) => (uint)_cuda.Blas.Max(GetDeviceVariable(segment), 1) - 1;

        public override bool IsEntirelyFinite(ITensorSegment2 segment) => _cuda.IsFinite(GetDeviceMemoryPtr(segment), segment.Size);

        public override ITensorSegment2 Reverse(ITensorSegment2 segment)
        {
            var ret = _cuda.Reverse(GetDeviceMemoryPtr(segment), segment.Size);
            return new CudaTensorSegment(ret);
        }

        public override float MeanSquaredDistance(ITensorSegment2 tensor, ITensorSegment2 other)
        {
            var size = GetSize(tensor, other);
            var temp = Subtract(tensor, other);
            try {
                var norm = temp.L2Norm();
                return norm * norm / size;
            }
            finally {
                temp.Release();
            }
        }

        public override float SquaredEuclideanDistance(ITensorSegment2 tensor, ITensorSegment2 other)
        {
            var temp = Subtract(tensor, other);
            try {
                var norm = L2Norm(temp);
                return norm * norm;
            }
            finally {
                temp.Release();
            }
        }

        public override float EuclideanDistance(ITensorSegment2 tensor, ITensorSegment2 other)
        {
            var size = GetSize(tensor, other);
            return _cuda.EuclideanDistance(GetDeviceMemoryPtr(tensor), GetDeviceMemoryPtr(other), size);
        }

        public override float ManhattanDistance(ITensorSegment2 tensor, ITensorSegment2 other)
        {
            var size = GetSize(tensor, other);
            return _cuda.ManhattanDistance(GetDeviceMemoryPtr(tensor), GetDeviceMemoryPtr(other), size);
        }

        public override float Average(ITensorSegment2 segment) => _cuda.SumValues(GetDeviceMemoryPtr(segment), segment.Size) / segment.Size;
        public override ITensorSegment2 Abs(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.Abs(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 Log(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.Log(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 Squared(ITensorSegment2 tensor) => PointwiseMultiply(tensor, tensor);
        public override ITensorSegment2 Sigmoid(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.Sigmoid(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 SigmoidDerivative(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.SigmoidDerivative(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 Tanh(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.TanH(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 TanhDerivative(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.TanHDerivative(GetDeviceMemoryPtr(tensor), tensor.Size));

        public override ITensorSegment2 Exp(ITensorSegment2 tensor)
        {
            return base.Exp(tensor);
        }

        public override ITensorSegment2 Relu(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.Relu(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 ReluDerivative(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.ReluDerivative(GetDeviceMemoryPtr(tensor), tensor.Size));
        public override ITensorSegment2 LeakyRelu(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.LeakyRelu(GetDeviceMemoryPtr(tensor), tensor.Size));

        public override ITensorSegment2 Softmax(ITensorSegment2 tensor)
        {
            var max = GetMax(tensor);

            var softmax = _cuda.SoftmaxVector(GetDeviceMemoryPtr(tensor), tensor.Size, max);
            var softmaxSum = _cuda.SumValues(softmax, tensor.Size);
            if (FloatMath.IsNotZero(softmaxSum))
                _cuda.Blas.Scale(1f / softmaxSum, softmax.DeviceVariable, 1);
            return new CudaTensorSegment(softmax);
        }

        public override IMatrix SoftmaxDerivative(ITensorSegment2 tensor)
        {
            var ret = _cuda.VectorSoftmaxDerivative(GetDeviceMemoryPtr(tensor), tensor.Size);
            return CreateMatrix(new CudaTensorSegment(ret), tensor.Size, tensor.Size);
        }

        public override ITensorSegment2 Pow(ITensorSegment2 tensor, float power) => new CudaTensorSegment(_cuda.Pow(GetDeviceMemoryPtr(tensor), tensor.Size, power));
        public override ITensorSegment2 Sqrt(ITensorSegment2 tensor) => new CudaTensorSegment(_cuda.Sqrt(GetDeviceMemoryPtr(tensor), tensor.Size, 1e-8f));
        public override void PointwiseDivideInPlace(ITensorSegment2 target, ITensorSegment2 other)
        {
            var size = GetSize(target, other);
            var ptr = GetDeviceMemoryPtr(target);
            var temp = _cuda.PointwiseDivide(ptr, GetDeviceMemoryPtr(other), size);
            ptr.CopyToDevice(temp);
            temp.Release();
        }

        public override void RoundInPlace(ITensorSegment2 tensor, float lower, float upper, float? mid)
        {
            _cuda.RoundInPlace(GetDeviceMemoryPtr(tensor), tensor.Size, lower, upper, mid ?? lower + (upper - lower) / 2);
        }

        public override void PointwiseMultiplyInPlace(ITensorSegment2 target, ITensorSegment2 other)
        {
            var size = GetSize(target, other);
            var ptr = GetDeviceMemoryPtr(target);
            var temp = _cuda.PointwiseMultiply(ptr, GetDeviceMemoryPtr(other), size);
            ptr.CopyToDevice(temp);
            temp.Release();
        }

        public override void MultiplyInPlace(ITensorSegment2 target, float scalar)
        {
            _cuda.Blas.Scale(scalar, GetDeviceVariable(target), 1);
        }

        public override IEnumerable<ITensorSegment2> Split(ITensorSegment2 segment, uint blockCount)
        {
            var blockSize = segment.Size / blockCount;
            var ptr = GetDeviceMemoryPtr(segment);

            return blockCount.AsRange().Select(i => {
                var ptr2 = _cuda.OffsetByBlock(ptr, i, blockSize);
                return new CudaTensorSegment(ptr2);
            });
        }

        public override float StdDev(ITensorSegment2 tensor, float? mean)
        {
            return _cuda.FindStdDev(GetDeviceMemoryPtr(tensor), tensor.Size, mean ?? Average(tensor));
        }

        public override uint? Search(ITensorSegment2 segment, float value)
        {
            return base.Search(segment, value);
        }

        public override IVector ColumnSums(IMatrix matrix)
        {
            var ret = _cuda.SumColumns(GetDeviceMemoryPtr(matrix.Segment), matrix.RowCount, matrix.ColumnCount);
            return CreateVector(new CudaTensorSegment(ret));
        }

        public override IVector GetDiagonal(IMatrix matrix)
        {
            var ret = _cuda.Diagonal(GetDeviceMemoryPtr(matrix.Segment), matrix.RowCount, matrix.ColumnCount);
            return CreateVector(new CudaTensorSegment(ret));
        }

        public override IMatrix Multiply(IMatrix matrix, IMatrix other)
        {
            var ret = _cuda.Allocate(matrix.RowCount * other.ColumnCount);
            int rowsA = (int)matrix.RowCount, columnsArowsB = (int)matrix.ColumnCount, columnsB = (int)other.ColumnCount;

            float alpha = 1.0f, beta = 0.0f;
            CudaBlasNativeMethods.cublasSgemm_v2(_cuda.Blas.CublasHandle,
                Operation.NonTranspose,
                Operation.NonTranspose,
                rowsA,
                columnsB,
                columnsArowsB,
                ref alpha,
                GetDeviceMemoryPtr(matrix.Segment).DevicePointer,
                rowsA,
                GetDeviceMemoryPtr(other.Segment).DevicePointer,
                columnsArowsB,
                ref beta,
                ret.DevicePointer,
                rowsA
            );
            return CreateMatrix(new CudaTensorSegment(ret), matrix.RowCount, other.ColumnCount);
        }

        public override IVector RowSums(IMatrix matrix)
        {
            var ret = _cuda.SumRows(GetDeviceMemoryPtr(matrix.Segment), matrix.RowCount, matrix.ColumnCount);
            return CreateVector(new CudaTensorSegment(ret));
        }

        public override IMatrix Transpose(IMatrix matrix)
        {
            var ret = _cuda.Allocate(matrix.RowCount * matrix.ColumnCount);
            float alpha = 1.0f, beta = 0.0f;
            CudaBlasNativeMethods.cublasSgeam(_cuda.Blas.CublasHandle,
                Operation.Transpose,
                Operation.NonTranspose,
                (int)matrix.RowCount,
                (int)matrix.ColumnCount,
                ref alpha,
                GetDeviceMemoryPtr(matrix.Segment).DevicePointer,
                (int)matrix.ColumnCount,
                ref beta,
                new CUdeviceptr(0),
                (int)matrix.ColumnCount,
                ret.DevicePointer,
                (int)matrix.RowCount
            );
            return CreateMatrix(new CudaTensorSegment(ret), matrix.ColumnCount, matrix.RowCount);
        }

        public override IMatrix TransposeSecondAndMultiply(IMatrix matrix, IMatrix other)
        {
            var ret = _cuda.Allocate(matrix.RowCount * other.RowCount);
            int rowsA = (int)matrix.RowCount, columnsArowsB = (int)matrix.ColumnCount, rowsB = (int)other.RowCount;

            float alpha = 1.0f, beta = 0.0f;
            CudaBlasNativeMethods.cublasSgemm_v2(_cuda.Blas.CublasHandle,
                Operation.NonTranspose,
                Operation.Transpose,
                rowsA,
                rowsB,
                columnsArowsB,
                ref alpha,
                GetDeviceMemoryPtr(matrix.Segment).DevicePointer,
                rowsA,
                GetDeviceMemoryPtr(other.Segment).DevicePointer,
                rowsB,
                ref beta,
                ret.DevicePointer,
                rowsA
            );
            return CreateMatrix(new CudaTensorSegment(ret), matrix.RowCount, other.RowCount);
        }

        public override IMatrix TransposeFirstAndMultiply(IMatrix matrix, IMatrix other)
        {
            var ret = _cuda.Allocate(matrix.ColumnCount * other.ColumnCount);
            int rowsA = (int)matrix.RowCount, columnsA = (int)matrix.ColumnCount, columnsB = (int)other.ColumnCount, rowsB = (int)other.RowCount;

            float alpha = 1.0f, beta = 0.0f;
            CudaBlasNativeMethods.cublasSgemm_v2(_cuda.Blas.CublasHandle,
                Operation.Transpose,
                Operation.NonTranspose,
                columnsA,
                columnsB,
                rowsB,
                ref alpha,
                GetDeviceMemoryPtr(matrix.Segment).DevicePointer,
                rowsA,
                GetDeviceMemoryPtr(other.Segment).DevicePointer,
                rowsB,
                ref beta,
                ret.DevicePointer,
                columnsA
            );
            return CreateMatrix(new CudaTensorSegment(ret), matrix.ColumnCount, other.ColumnCount);
        }

        public override (IMatrix U, IVector S, IMatrix VT) Svd(IMatrix matrix)
        {
            var solver = _cuda.Solver;
            var rows = matrix.RowCount;
            var columns = matrix.ColumnCount;

            // find the size of the required buffer
            var bufferSize = solver.GesvdBufferSizeFloat((int)rows, (int)columns);
            var mn = System.Math.Min(rows, columns);

            // allocate output buffers
            var s = _cuda.Allocate(mn);
            var u = _cuda.Allocate(rows * rows);
            var vt = _cuda.Allocate(columns * columns);

            // call cusolver to find the SVD
            try {
                var buffer = _cuda.Allocate((uint)bufferSize);
                var rwork = _cuda.Allocate(mn);
                var a = _cuda.Allocate(rows * columns);
                try {
                    using var devInfo = new CudaDeviceVariable<int>(1);
                    a.CopyToDevice(GetDeviceMemoryPtr(matrix.Segment));
                    solver.Gesvd(
                        'A', 
                        'A', 
                        (int)rows, 
                        (int)columns, 
                        a.DeviceVariable, 
                        (int)rows, 
                        s.DeviceVariable, 
                        u.DeviceVariable, 
                        (int)rows, 
                        vt.DeviceVariable, 
                        (int)columns, 
                        buffer.DeviceVariable, 
                        bufferSize, 
                        rwork.DeviceVariable, 
                        devInfo
                    );
                    return (
                        CreateMatrix(new CudaTensorSegment(u), rows, rows),
                        CreateVector(new CudaTensorSegment(s)),
                        CreateMatrix(new CudaTensorSegment(vt), columns, columns)
                    );
                }finally {
                    buffer.Release();
                    rwork.Release();
                    a.Release();
                }
            }catch {
                s.Release();
                u.Release();
                vt.Release();
                throw;
            }
        }

        public override ITensorSegment2 CherryPickIndices(ITensorSegment2 tensor, uint[] indices)
        {
            var data = _cuda.VectorCopy(GetDeviceMemoryPtr(tensor), indices);
            return new CudaTensorSegment(data);
        }

        public override void EuclideanNormalization(ITensorSegment2 segment)
        {
            var norm = L2Norm(segment);
            if (FloatMath.IsNotZero(norm))
                Multiply(segment, 1f / norm);
        }

        public override void FeatureScaleNormalization(ITensorSegment2 segment)
        {
            var ptr = GetDeviceMemoryPtr(segment);
            var (min, max) = _cuda.FindMinAndMax(ptr, segment.Size);
            var range = max - min;
            if (FloatMath.IsNotZero(range))
                _cuda.Normalise(ptr, segment.Size, min, range);
        }

        public override void ManhattanNormalization(ITensorSegment2 segment)
        {
            var norm = L1Norm(segment);
            if (FloatMath.IsNotZero(norm))
                Multiply(segment, 1f / norm);
        }

        public override void StandardNormalization(ITensorSegment2 segment)
        {
            var mean = Average(segment);
            var stdDev = StdDev(segment, mean);
            if (FloatMath.IsNotZero(stdDev))
                _cuda.Normalise(GetDeviceMemoryPtr(segment), segment.Size, mean, stdDev);
        }

        public override (IMatrix Left, IMatrix Right) SplitAtColumn(IMatrix matrix, uint columnIndex)
        {
            var ret1 = CreateMatrix(matrix.RowCount, columnIndex, (x, y) => matrix[x, y]);
            var ret2 = CreateMatrix(matrix.RowCount, matrix.ColumnCount - columnIndex, (x, y) => matrix[x, columnIndex + y]);
            return (ret1, ret2);
        }

        public override (IMatrix Top, IMatrix Bottom) SplitAtRow(IMatrix matrix, uint rowIndex)
        {
            var size = matrix.RowCount - rowIndex;
            var ret1 = _cuda.Allocate(rowIndex * matrix.ColumnCount);
            var ret2 = _cuda.Allocate(size * matrix.ColumnCount);
            _cuda.SplitColumns(GetDeviceMemoryPtr(matrix.Segment), ret1, ret2, matrix.RowCount, matrix.ColumnCount, rowIndex);
            return (CreateMatrix(new CudaTensorSegment(ret1), rowIndex, matrix.ColumnCount), CreateMatrix(new CudaTensorSegment(ret2), size, matrix.ColumnCount));
        }

        public override IMatrix ConcatColumns(IMatrix top, IMatrix bottom)
        {
            Debug.Assert(top.ColumnCount == bottom.ColumnCount);
            var size = top.RowCount + bottom.RowCount;
            var ret = _cuda.Allocate(size * top.ColumnCount);
            _cuda.ConcatColumns(GetDeviceMemoryPtr(top.Segment), GetDeviceMemoryPtr(bottom.Segment), ret, size, top.ColumnCount, top.RowCount, bottom.RowCount);
            return CreateMatrix(new CudaTensorSegment(ret), size, top.ColumnCount);
        }

        public override IMatrix ConcatRows(IMatrix left, IMatrix right)
        {
            Debug.Assert(left.RowCount == right.RowCount);
            var size = left.ColumnCount + right.ColumnCount;
            var ret = _cuda.Allocate(left.RowCount * size);
            _cuda.ConcatRows(GetDeviceMemoryPtr(left.Segment), GetDeviceMemoryPtr(right.Segment), ret, left.RowCount, size, left.ColumnCount);
            return CreateMatrix(new CudaTensorSegment(ret), left.RowCount, size);
        }

        static CudaDeviceVariable<float> GetDeviceVariable(ITensorSegment2 segment) => GetDeviceMemoryPtr(segment).DeviceVariable;
        static IDeviceMemoryPtr GetDeviceMemoryPtr(ITensorSegment2 segment)
        {
            if (segment is CudaTensorSegment cudaSegment) {
                if (!segment.IsValid)
                    throw new Exception("CUDA tensor was not valid");
                return cudaSegment.DeviceMemory;
            }

            throw new Exception("CUDA tensors can only be used with other CUDA tensors");
        }
    }
}