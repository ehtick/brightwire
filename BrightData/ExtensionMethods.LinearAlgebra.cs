﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using BrightData.Helper;
using BrightData.LinearAlgebra;
using BrightData.LinearAlgebra.ReadOnly;
using CommunityToolkit.HighPerformance.Buffers;
using CommunityToolkit.HighPerformance.Helpers;

namespace BrightData
{
    public static partial class ExtensionMethods
    {
        /// <summary>
        /// Creates a vector
        /// </summary>
        /// <param name="_"></param>
        /// <param name="size">Size of vector</param>
        /// <param name="initializer">Callback to initialize each value (optional)</param>
        /// <returns></returns>
        public static IReadOnlyVector CreateReadOnlyVector(this BrightDataContext _, uint size, Func<uint, float>? initializer) => initializer is not null
            ? new ReadOnlyVector(size, initializer)
            : new ReadOnlyVector(size)
        ;


        /// <summary>
        /// Creates a vector
        /// </summary>
        /// <param name="context"></param>
        /// <param name="size">Size of vector</param>
        /// <param name="initializer">Callback to initialize each value (optional)</param>
        /// <returns></returns>
        public static IReadOnlyVector CreateReadOnlyVector(this BrightDataContext context, int size, Func<uint, float>? initializer) => CreateReadOnlyVector(context, (uint)size, initializer);

        /// <summary>
        /// Creates a vector
        /// </summary>
        /// <param name="_"></param>
        /// <param name="size">Size of vector</param>
        /// <param name="initialValue">Initial value of each element</param>
        /// <returns></returns>
        public static IReadOnlyVector CreateReadOnlyVector(this BrightDataContext _, uint size, float initialValue = 0f) => initialValue == 0f
            ? new ReadOnlyVector(size)
            : new ReadOnlyVector(size, _ => initialValue)
        ;

        /// <summary>
        /// Creates a vector
        /// </summary>
        /// <param name="context"></param>
        /// <param name="size">Size of vector</param>
        /// <param name="initialValue">Initial value of each element</param>
        /// <returns></returns>
        public static IReadOnlyVector CreateReadOnlyVector(this BrightDataContext context, int size, float initialValue = 0f) => CreateReadOnlyVector(context, (uint)size, initialValue);

        /// <summary>
        /// Creates a vector
        /// </summary>
        /// <param name="_"></param>
        /// <param name="initialData">Initial data</param>
        /// <returns></returns>
        public static IReadOnlyVector CreateReadOnlyVector(this BrightDataContext _, params float[] initialData) => new ReadOnlyVector(initialData);

        /// <summary>
        /// Creates a vector from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static IReadOnlyVector CreateReadOnlyVector(this BrightDataContext context, BinaryReader reader)
        {
            var ret = GenericActivator.CreateUninitialized<ICanInitializeFromBinaryReader>(typeof(ReadOnlyVector));
            ret.Initialize(context, reader);
            return (IReadOnlyVector)ret;
        }


        /// <summary>
        /// Creates a matrix
        /// </summary>
        /// <param name="_"></param>
        /// <param name="rows">Number of rows</param>
        /// <param name="columns">Number of columns</param>
        /// <param name="initializer">Callback to initialize each value (optional)</param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrix(this BrightDataContext _, uint rows, uint columns, Func<uint, uint, float>? initializer = null) => initializer is not null
            ? new ReadOnlyMatrix(rows, columns, initializer)
            : new ReadOnlyMatrix(rows, columns)
        ;

        /// <summary>
        /// Creates a matrix
        /// </summary>
        /// <param name="_"></param>
        /// <param name="rows">Number of rows</param>
        /// <param name="columns">Number of columns</param>
        /// <param name="initialValue">Initial value of each element</param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrix(this BrightDataContext _, uint rows, uint columns, float initialValue = 0f) => initialValue == 0f
            ? new ReadOnlyMatrix(rows, columns)
            : new ReadOnlyMatrix(rows, columns, (_, _) => initialValue)
        ;

        /// <summary>
        /// Creates a matrix from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrix(this BrightDataContext context, BinaryReader reader)
        {
            var ret = GenericActivator.CreateUninitialized<ICanInitializeFromBinaryReader>(typeof(ReadOnlyMatrix));
            ret.Initialize(context, reader);
            return (IReadOnlyMatrix)ret;
        }

        /// <summary>
        /// Creates a matrix from vectors (each will become a row)
        /// </summary>
        /// <param name="_"></param>
        /// <param name="rows"></param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrixFromRows(this BrightDataContext _, params IReadOnlyVector[] rows)
        {
            var columns = rows[0].Size;
            var ret = new ReadOnlyMatrix((uint)rows.Length, columns);
            for (var i = 0; i < rows.Length; i++) {
                var source = rows[i];
                var target = ret.GetRow((uint)i);
                source.Segment.CopyTo(target.Segment);
            } 
            return ret;
        }

        /// <summary>
        /// Creates a matrix from rows (each will become a row)
        /// </summary>
        /// <param name="_"></param>
        /// <param name="rows"></param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrixFromRows(this BrightDataContext _, params float[][] rows)
        {
            var columns = (uint)rows[0].Length;
            var ret = new ReadOnlyMatrix((uint)rows.Length, columns);
            for (var i = 0; i < rows.Length; i++) {
                var source = rows[i];
                var target = ret.GetRow((uint)i);
                target.Segment.CopyFrom(source);
            } 
            return ret;
        }

        /// <summary>
        /// Creates a matrix from vectors (each will become a column)
        /// </summary>
        /// <param name="_"></param>
        /// <param name="columns"></param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrixFromColumns(this BrightDataContext _, params IReadOnlyVector[] columns)
        {
            var rows = columns[0].Size;
            var ret = new ReadOnlyMatrix(rows, (uint)columns.Length);
            for (var i = 0; i < columns.Length; i++) {
                var source = columns[i];
                var target = ret.GetRow((uint)i);
                source.Segment.CopyTo(target.Segment);
            } 
            return ret;
        }

        /// <summary>
        /// Creates a matrix from vectors (each will become a column)
        /// </summary>
        /// <param name="_"></param>
        /// <param name="columns"></param>
        /// <returns></returns>
        public static IReadOnlyMatrix CreateReadOnlyMatrixFromColumns(this BrightDataContext _, params float[][] columns)
        {
            var rows = (uint)columns[0].Length;
            var ret = new ReadOnlyMatrix(rows, (uint)columns.Length);
            for (var i = 0; i < columns.Length; i++) {
                var source = columns[i];
                var target = ret.GetRow((uint)i);
                target.Segment.CopyFrom(source);
            } 
            return ret;
        }

        /// <summary>
        /// Creates a 3D tensor from matrices
        /// </summary>
        /// <param name="_"></param>
        /// <param name="matrices"></param>
        /// <returns></returns>
        public static IReadOnlyTensor3D CreateReadOnlyTensor3D(this BrightDataContext _, params IReadOnlyMatrix[] matrices) => new ReadOnlyTensor3D(matrices);

        /// <summary>
        /// Create a 3D tensor from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static IReadOnlyTensor3D CreateReadOnlyTensor3D(this BrightDataContext context, BinaryReader reader)
        {
            var ret = GenericActivator.CreateUninitialized<ICanInitializeFromBinaryReader>(typeof(ReadOnlyTensor3D));
            ret.Initialize(context, reader);
            return (IReadOnlyTensor3D)ret;
        }

        /// <summary>
        /// Creates a 4D tensor from matrices
        /// </summary>
        /// <param name="_"></param>
        /// <param name="tensors"></param>
        /// <returns></returns>
        public static IReadOnlyTensor4D CreateReadOnlyTensor4D(this BrightDataContext _, params IReadOnlyTensor3D[] tensors) => new ReadOnlyTensor4D(tensors);

        /// <summary>
        /// Creates a 4D tensor from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static IReadOnlyTensor4D CreateReadOnlyTensor4D(this BrightDataContext context, BinaryReader reader)
        {
            var ret = GenericActivator.CreateUninitialized<ICanInitializeFromBinaryReader>(typeof(ReadOnlyTensor4D));
            ret.Initialize(context, reader);
            return (IReadOnlyTensor4D)ret;
        }

        /// <summary>
        /// Creates an identity matrix (each diagonal element is 1, each other element is 0)
        /// </summary>
        /// <param name="lap"></param>
        /// <param name="size">Width and height of the new matrix</param>
        /// <returns></returns>
        public static IMatrix CreateIdentityMatrix(this LinearAlgebraProvider lap, uint size)
        {
            return lap.CreateMatrix(size, size, (x, y) => x == y ? 1f : 0f);
        }

        /// <summary>
        /// Creates a diagonal matrix
        /// </summary>
        /// <param name="lap"></param>
        /// <param name="values">Diagonal values</param>
        /// <returns></returns>
        public static IMatrix CreateDiagonalMatrix(this LinearAlgebraProvider lap, params float[] values)
        {
            return lap.CreateMatrix((uint)values.Length, (uint)values.Length, (x, y) => x == y ? values[x] : 0f);
        }

        /// <summary>
        /// Randomly initialize a tensor
        /// </summary>
        /// <param name="tensor"></param>
        public static void InitializeRandomly(this ITensor tensor)
        {
            var segment = tensor.Segment;
            var context = tensor.Context;
            for (uint i = 0, len = segment.Size; i < len; i++)
                segment[i] = context.NextRandomFloat();
        }

        /// <summary>
        /// Initialize a tensor to a single value
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="value">Value to initialize each element of the tensor</param>
        public static void Initialize(this ITensor tensor, float value)
        {
            var segment = tensor.Segment;
            for (uint i = 0, len = segment.Size; i < len; i++)
                segment[i] = value;
        }

        /// <summary>
        /// Initialize a tensor using a callback
        /// </summary>
        /// <param name="tensor"></param>
        /// <param name="initializer">Callback for each element</param>
        public static void Initialize(this ITensor tensor, Func<uint, float> initializer)
        {
            var segment = tensor.Segment;
            for (uint i = 0, len = segment.Size; i < len; i++)
                segment[i] = initializer(i);
        }

        /// <summary>
        /// Converts the tensor segment to a sparse format (only non zero entries are preserved)
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        public static WeightedIndexList ToSparse(this ITensorSegment segment)
        {
            return WeightedIndexList.Create(segment.Values
                .Select((v, i) => new WeightedIndexList.Item((uint)i, v))
                .Where(d => FloatMath.IsNotZero(d.Weight))
            );
        }

        /// <summary>
        /// Reads a vector from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static IReadOnlyVector LoadReadOnlyVectorFrom(this BrightDataContext context, BinaryReader reader)
        {
            if (context.Get(Consts.LegacyFloatSerialisationInput, false))
            {
                var len = reader.ReadInt32();
                var ret = new float[len];
                for (var i = 0; i < len; i++)
                    ret[i] = reader.ReadSingle();
                return context.CreateReadOnlyVector(ret);
            }
            return context.CreateReadOnlyVector(reader);
        }

        /// <summary>
        /// Reacts a float array from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static float[] LoadReadOnlyVectorAndThenGetArrayFrom(this BrightDataContext context, BinaryReader reader)
        {
            if (context.Get(Consts.LegacyFloatSerialisationInput, false))
            {
                var len = reader.ReadInt32();
                var ret = new float[len];
                for (var i = 0; i < len; i++)
                    ret[i] = reader.ReadSingle();
                return ret;
            }
            // TODO: refactor this to avoid creating the vector?
            var temp = context.CreateReadOnlyVector(reader);
            return temp.Segment.ToNewArray();
        }

        /// <summary>
        /// Reads a matrix from a binary reader
        /// </summary>
        /// <param name="context"></param>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static IReadOnlyMatrix ReadMatrixFrom(this BrightDataContext context, BinaryReader reader)
        {
            if (context.Get(Consts.LegacyFloatSerialisationInput, false))
            {
                var len = reader.ReadInt32();
                var ret = new IReadOnlyVector[len];
                for (var i = 0; i < len; i++)
                    ret[i] = context.LoadReadOnlyVectorFrom(reader);
                return context.CreateReadOnlyMatrixFromRows(ret);
            }
            return context.CreateReadOnlyMatrix(reader);
        }

        /// <summary>
        /// Reduce dimensions of the matrix with a singular value decomposition
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="dimensions">Number of dimensions to reduce to</param>
        /// <returns></returns>
        public static IMatrix ReduceDimensionsWithSvd(this IMatrix matrix, uint dimensions)
        {
            using var matrixT = matrix.Transpose();
            var (u, vector, vt) = matrixT.Svd();

            try {
                using var s = matrix.LinearAlgebraProvider.CreateDiagonalMatrix(vector.Segment.Values.Take((int)dimensions).ToArray());
                using var v2 = vt.GetNewMatrixFromRows(dimensions.AsRange());
                return s.Multiply(v2);
            }
            finally {
                u.Dispose();
                vector.Dispose();
                vt.Dispose();
            }
        }

        /// <summary>
        /// Calculates an average from a collection of tensors
        /// </summary>
        /// <param name="tensors">Tensors to average</param>
        /// <param name="dispose">True to dispose each of the input vectors</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static T Average<T>(this IEnumerable<T> tensors, bool dispose)
            where T: ITensor
        {
            if(tensors == null)
                throw new ArgumentException("Null enumerable", nameof(tensors));
            ITensorSegment? ret = null;
            try {
                var count = 0;
                LinearAlgebraProvider? lap = null;
                uint[]? shape = null;
                foreach (var item in tensors) {
                    if (ret is null) {
                        ret = (lap ??= item.LinearAlgebraProvider).CreateSegment(item.TotalSize, false);
                        item.Segment.CopyTo(ret);
                        shape = item.Shape;
                    }
                    else
                        ret.AddInPlace(item.Segment);

                    ++count;
                    if (dispose)
                        item.Dispose();
                }

                if (ret is null || lap is null || shape is null)
                    throw new ArgumentException("Empty enumerable", nameof(tensors));
                ret.MultiplyInPlace(1f / count);
                return (T)lap.CreateTensor(shape, ret);
            }
            catch {
                ret?.Dispose();
                throw;
            }
        }

        /// <summary>
        /// Returns an array from a tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]public static float[] GetLocalOrNewArray(this ITensorSegment segment) => segment.GetArrayIfEasilyAvailable() ?? segment.ToNewArray();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static MemoryOwner<float> Allocate(uint size) => MemoryOwner<float>.Allocate((int)size);

        /// <summary>
        /// Creates a new tensor segment from each pairwise combination of this and another tensor segment (in parallel)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <param name="func">Pairwise combiner function</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment ZipParallel(this ITensorSegment segment, ITensorSegment other, Func<float, float, float> func)
        {
            var size = segment.Size;
            if (size != other.Size)
                throw new ArgumentException("Segments were different sizes");

            var ret = Allocate(size);
            var array = ret.DangerousGetArray().Array!;
            if (segment.Size >= Consts.MinimumSizeForParallel)
                ParallelHelper.For(0, (int)size, new ZipAction(segment, other, func, array));
            else {
                for (uint i = 0; i < size; i++)
                    array[(int)i] = func(segment[i], other[i]);
            }

            return new ArrayPoolTensorSegment(ret);
        }

        /// <summary>
        /// Creates a new tensor segment from each pairwise combination of this and another tensor segment (vectorized)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <param name="func1">Vectorized combiner combiner</param>
        /// <param name="func2">Pairwise combiner function</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment ZipVectorized(
            this ITensorSegment segment,
            ITensorSegment other,
            ComputeVectorisedTwo<float> func1,
            Func<float, float, float> func2)
        {
            var size = segment.Size;
            if (size != other.Size)
                throw new ArgumentException("Segments were different sizes");

            MemoryOwner<float> ret;
            if (size >= Consts.MinimumSizeForVectorised) {
                SpanOwner<float> leftTemp = SpanOwner<float>.Empty, rightTemp = SpanOwner<float>.Empty;
                var leftPtr = segment.GetSpan(ref leftTemp, out var wasLefTempUsed);
                var rightPtr = other.GetSpan(ref rightTemp, out var wasRightTempUsed);
                try {
                    ret = leftPtr.ZipVectorized(rightPtr, func1, func2);
                }
                finally {
                    if (wasLefTempUsed)
                        leftTemp.Dispose();
                    if (wasRightTempUsed)
                        rightTemp.Dispose();
                }
            }
            else {
                ret = Allocate(size);
                var retPtr = ret.Span;
                for (var i = 0; i < size; i++)
                    retPtr[i] = func2(segment[i], other[i]);
            }

            return new ArrayPoolTensorSegment(ret);
        }

        /// <summary>
        /// Creates a new tensor segment from this tensor segment (in parallel)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="transformer">Value transformer</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment TransformParallel(this ITensorSegment segment, Func<float, float> transformer)
        {
            var size = segment.Size;
            var ret = Allocate(size);
            var array = ret.DangerousGetArray().Array!;

            if (size >= Consts.MinimumSizeForParallel)
                ParallelHelper.For(0, (int)size, new TransformAction(segment, transformer, array));
            else {
                for (uint i = 0; i < size; i++)
                    array[(int)i] = transformer(segment[i]);
            }

            return new ArrayPoolTensorSegment(ret);
        }

        /// <summary>
        /// Creates a new tensor segment from this tensor segment (vectorized)
        /// </summary>
        /// <param name="segment">This segment</param>
        /// <param name="transformer1">Vectorized value transformer</param>
        /// <param name="transformer2">Value transformer</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment TransformVectorized(
            this ITensorSegment segment,
            ComputeVectorisedOne<float> transformer1,
            Func<float, float> transformer2)
        {
            var size = segment.Size;
            MemoryOwner<float> ret;

            if (size >= Consts.MinimumSizeForVectorised) {
                var leftTemp = SpanOwner<float>.Empty;
                var leftPtr = segment.GetSpan(ref leftTemp, out var wasLeftTempUsed);
                try {
                    ret = leftPtr.TransformVectorized(transformer1, transformer2);
                }
                finally {
                    if (wasLeftTempUsed)
                        leftTemp.Dispose();
                }
            }
            else {
                ret = Allocate(segment.Size);
                var resultPtr = ret.Span;
                for (var i = 0; i < size; i++)
                    resultPtr[i] = transformer2(segment[i]);
            }

            return new ArrayPoolTensorSegment(ret);
        }

        
        /// <summary>
        /// Creates a new tensor segment from this tensor segment (in parallel)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="transformer">Indexed transformer</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]public static ITensorSegment TransformParallelIndexed(this ITensorSegment segment, Func<uint, float> transformer)
        {
            var size = segment.Size;
            var ret = Allocate(size);
            var array = ret.DangerousGetArray().Array!;

            if (size >= Consts.MinimumSizeForParallel)
                ParallelHelper.For(0, (int)size, new TransformIndexedAction(transformer, array));
            else {
                for (uint i = 0; i < size; i++)
                    array[(int)i] = transformer(i);
            }

            return new ArrayPoolTensorSegment(ret);
        }

        /// <summary>
        /// In place update of this tensor segment with pairwise values from another tensor segment (in parallel)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <param name="func">Update function</param>
        /// <exception cref="ArgumentException"></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MutateParallel(this ITensorSegment segment, ITensorSegment other, Func<float, float, float> func)
        {
            var size = segment.Size;
            if (size != other.Size)
                throw new ArgumentException("Segments were different sizes");

            if (size >= Consts.MinimumSizeForParallel)
                Parallel.For(0, size, i => segment[i] = func(segment[i], other[i]));
            else {
                for (uint i = 0; i < size; i++)
                    segment[i] = func(segment[i], other[i]);
            }
        }

        /// <summary>
        /// In place update of this tensor segment with pairwise values from another tensor segment (vectorized)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <param name="func1">Vectorized update function</param>
        /// <param name="func2">Update function</param>
        /// <exception cref="ArgumentException"></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MutateVectorized(
            this ITensorSegment segment,
            ITensorSegment other,
            ComputeVectorisedTwo<float> func1,
            Func<float, float, float> func2)
        {
            var size = segment.Size;
            if (size != other.Size)
                throw new ArgumentException("Segments were different sizes");

            if (size >= Consts.MinimumSizeForVectorised) {
                // get pointers to the segments
                SpanOwner<float> leftTemp = SpanOwner<float>.Empty, rightTemp = SpanOwner<float>.Empty;
                var leftPtr = segment.GetSpan(ref leftTemp, out var wasLeftTempUsed);
                var rightPtr = other.GetSpan(ref rightTemp, out var wasRightTempUsed);
                try {
                    fixed (float* fp = &MemoryMarshal.GetReference(leftPtr)) {
                        var leftMutablePtr = new Span<float>(fp, (int)size);
                        leftMutablePtr.MutateVectorized(rightPtr, func1, func2);
                    }
                }
                finally {
                    if (wasLeftTempUsed)
                        leftTemp.Dispose();
                    if (wasRightTempUsed)
                        rightTemp.Dispose();
                }
            }
            else {
                for (var i = 0; i < size; i++)
                    segment[i] = func2(segment[i], other[i]);
            }
        }

        /// <summary>
        /// In place update of tensor segment (in parallel)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="mutator">Update function</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MutateInPlaceParallel(this ITensorSegment segment, Func<float, float> mutator)
        {
            var size = segment.Size;
            if (size >= Consts.MinimumSizeForParallel)
                ParallelHelper.For(0, (int)size, new MutateAction(mutator, segment));
            else {
                for (uint i = 0; i < size; i++)
                    segment[i] = mutator(segment[i]);
            }
        }

        /// <summary>
        /// In place update of tensor segment (vectorized)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="mutator1">Vectorized update function</param>
        /// <param name="mutator2">Update function</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe void MutateInPlaceVectorized(
            this ITensorSegment segment,
            ComputeVectorisedOne<float> mutator1,
            Func<float, float> mutator2)
        {
            var size = segment.Size;
            if (size >= Consts.MinimumSizeForVectorised) {
                var leftTemp = SpanOwner<float>.Empty;
                var leftPtr = segment.GetSpan(ref leftTemp, out var wasLeftTempUsed);
                try {
                    fixed (float* fp = &MemoryMarshal.GetReference(leftPtr)) {
                        var leftMutablePtr = new Span<float>(fp, (int)size);
                        leftMutablePtr.MutateInPlaceVectorized(mutator1, mutator2);
                    }
                }
                finally {
                    if (wasLeftTempUsed)
                        leftTemp.Dispose();
                }
            }
            else {
                for (var i = 0; i < size; i++)
                    segment[i] = mutator2(segment[i]);
            }
        }

        /// <summary>
        /// Sums all values
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sum(this ITensorSegment segment)
        {
            var size = segment.Size;
            if (size >= Consts.MinimumSizeForVectorised && Sse3.IsSupported) {
                var temp = SpanOwner<float>.Empty;
                var span = segment.GetSpan(ref temp, out var wasTempUsed);
                try {
                    return span.Sum();
                }
                finally {
                    if (wasTempUsed)
                        temp.Dispose();
                }
            }

            var ret = 0f;
            for (uint i = 0; i < size; i++)
                ret += segment[i];
            return ret;
        }

        /// <summary>
        /// Pairwise addition of this with another tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor1">This tensor</param>
        /// <param name="tensor2">Other tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Add(this ITensorSegment tensor1, ITensorSegment tensor2) => ZipVectorized(
            tensor1,
            tensor2,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a + b,
            (a, b) => a + b
        );

        /// <summary>
        /// Pairwise addition of this with another tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor1">This tensor</param>
        /// <param name="tensor2">Other tensor</param>
        /// <param name="coefficient1">Value to multiply each value in this tensor</param>
        /// <param name="coefficient2">Value to multiply each value in the other tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Add(this ITensorSegment tensor1, ITensorSegment tensor2, float coefficient1, float coefficient2) => ZipVectorized(
            tensor1,
            tensor2,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a * coefficient1 + b * coefficient2,
            (a, b) => a * coefficient1 + b * coefficient2
        );

        /// <summary>
        /// Adds a scalar to each value in this tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="scalar">Scalar to add to each value</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Add(this ITensorSegment tensor, float scalar)
        {
            var scalarVector = new Vector<float>(scalar);
            return TransformVectorized(
                tensor,
                (in Vector<float> a, out Vector<float> r) => r = a + scalarVector,
                a => a + scalar
            );
        }

        /// <summary>
        /// Adds another tensor segment in place to this tensor segment
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="other">Other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(this ITensorSegment target, ITensorSegment other) => MutateVectorized(
            target,
            other,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a + b,
            (a, b) => a + b
        );

        /// <summary>
        /// Adds another tensor segment in place to this tensor segment
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <param name="coefficient1">Value to multiply each value in this tensor</param>
        /// <param name="coefficient2">Value to multiply each value in the other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(this ITensorSegment target, ITensorSegment other, float coefficient1, float coefficient2) => MutateVectorized(
            target,
            other,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = (a * coefficient1) + (b * coefficient2),
            (a, b) => (a * coefficient1) + (b * coefficient2)
        );

        /// <summary>
        /// Adds a scalar to each value in this tensor segment in place
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="scalar">Scalar to add to each value</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void AddInPlace(this ITensorSegment target, float scalar)
        {
            var scalarVector = new Vector<float>(scalar);
            MutateInPlaceVectorized(
                target,
                (in Vector<float> a, out Vector<float> r) => r = a + scalarVector,
                a => a + scalar
            );
        }

        /// <summary>
        /// Multiplies each value in this tensor segment by a scalar (in place)
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="scalar">Scalar to multiply</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void MultiplyInPlace(this ITensorSegment target, float scalar)
        {
            var scalarVector = new Vector<float>(scalar);
            MutateInPlaceVectorized(
                target,
                (in Vector<float> a, out Vector<float> r) => r = a * scalarVector,
                a => a * scalar
            );
        }

        /// <summary>
        /// Multiplies each value in this tensor segment by a scalar into a new tensor segment
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="scalar">Scalar to multiply</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Multiply(this ITensorSegment target, float scalar)
        {
            var scalarVector = new Vector<float>(scalar);
            return TransformVectorized(
                target,
                (in Vector<float> a, out Vector<float> r) => r = a * scalarVector,
                a => a * scalar
            );
        }

        /// <summary>
        /// Subtracts another tensor segment from this tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor1">This tensor</param>
        /// <param name="tensor2">Other tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Subtract(this ITensorSegment tensor1, ITensorSegment tensor2) => ZipVectorized(
            tensor1,
            tensor2,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a - b,
            (a, b) => a - b
        );

        /// <summary>
        /// Subtracts another tensor segment from this tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor1">This tensor</param>
        /// <param name="tensor2">Other tensor</param>
        /// <param name="coefficient1">Value to multiply each value in this tensor</param>
        /// <param name="coefficient2">Value to multiply each value in the other tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Subtract(this ITensorSegment tensor1, ITensorSegment tensor2, float coefficient1, float coefficient2) => ZipVectorized(
            tensor1,
            tensor2,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a * coefficient1 - b * coefficient2,
            (a, b) => a * coefficient1 - b * coefficient2
        );

        /// <summary>
        /// Subtracts another tensor segment from this tensor segment in place
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="other">Other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SubtractInPlace(this ITensorSegment target, ITensorSegment other) => MutateVectorized(
            target,
            other,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a - b,
            (a, b) => a - b
        );

        /// <summary>
        /// Subtracts another tensor segment from this tensor segment in place
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <param name="coefficient1">Value to multiply each value in this tensor</param>
        /// <param name="coefficient2">Value to multiply each value in the other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void SubtractInPlace(this ITensorSegment target, ITensorSegment other, float coefficient1, float coefficient2) => MutateVectorized(
            target,
            other,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a * coefficient1 - b * coefficient2,
            (a, b) => a * coefficient1 - b * coefficient2
        );

        /// <summary>
        /// Pairwise multiply each value in this tensor segment with the corresponding value from another tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor1">This tensor</param>
        /// <param name="tensor2">Other tensor</param>
        /// <returns></returns>
        public static ITensorSegment PointwiseMultiply(this ITensorSegment tensor1, ITensorSegment tensor2) => ZipVectorized(
            tensor1,
            tensor2,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a * b,
            (a, b) => a * b
        );

        /// <summary>
        /// Pairwise multiply each value in this tensor segment with the corresponding value from another tensor segment in place
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="other">Other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void PointwiseMultiplyInPlace(this ITensorSegment target, ITensorSegment other) => MutateVectorized(
            target,
            other,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a * b,
            (a, b) => a * b
        );

        /// <summary>
        /// Pairwise divide each value in this tensor segment with the corresponding value from another tensor segment into a new tensor segment
        /// </summary>
        /// <param name="tensor1">This tensor</param>
        /// <param name="tensor2">Other tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment PointwiseDivide(this ITensorSegment tensor1, ITensorSegment tensor2) => ZipVectorized(
            tensor1,
            tensor2,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a / b,
            (a, b) => a / b
        );

        /// <summary>
        /// Pairwise divide each value in this tensor segment with the corresponding value from another tensor segment in place
        /// </summary>
        /// <param name="target">This tensor</param>
        /// <param name="other">Other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void PointwiseDivideInPlace(this ITensorSegment target, ITensorSegment other) => MutateVectorized(
            target,
            other,
            (in Vector<float> a, in Vector<float> b, out Vector<float> r) => r = a / b,
            (a, b) => a / b
        );

        /// <summary>
        /// Calculates the dot product of this with another tensor
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct(this ITensorSegment segment, ITensorSegment other)
        {
            var size = segment.Size;
            if (size != other.Size)
                throw new ArgumentException("Segments were different sizes");
            if (size >= Consts.MinimumSizeForVectorised) {
                SpanOwner<float> leftTemp = SpanOwner<float>.Empty, rightTemp = SpanOwner<float>.Empty;
                var leftPtr = segment.GetSpan(ref leftTemp, out var wasLefTempUsed);
                var rightPtr = other.GetSpan(ref rightTemp, out var wasRightTempUsed);
                try {
                    return leftPtr.DotProduct(rightPtr);
                }
                finally {
                    if (wasLefTempUsed)
                        leftTemp.Dispose();
                    if (wasRightTempUsed)
                        rightTemp.Dispose();
                }
            }

            var ret = 0f;
            for (uint i = 0; i < size; i++)
                ret += segment[i] * other[i];
            return ret;
        }

        /// <summary>
        /// Creates a new tensor segment that contains the square root of each value in this tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="adjustment">A small value to add to each value in case of zeros</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Sqrt(this ITensorSegment tensor, float adjustment = FloatMath.AlmostZero)
        {
            Vector<float> adjustmentVector = new(adjustment);
            return TransformVectorized(tensor,
                (in Vector<float> a, out Vector<float> r) => r = Vector.SquareRoot(a + adjustmentVector),
                x => MathF.Sqrt(x + adjustment)
            );
        }

        /// <summary>
        /// Searches this tensor segment for the index of the first value that matches the specified value within a level of tolerance
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="value">Value to find</param>
        /// <param name="tolerance">Degree of tolerance</param>
        /// <returns></returns>
        public static uint? Search(this ITensorSegment segment, float value, float tolerance = FloatMath.AlmostZero)
        {
            uint? ret = null;
            Analyze(segment, (v, index) => {
                if (Math.Abs(value - v) < tolerance)
                    ret = index;
            });
            return ret;
        }

        /// <summary>
        /// Constrains each value in this tensor segment to fit between a supplied minimum and maximum value
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="minInclusiveValue">Minimum allowed inclusive value (optional)</param>
        /// <param name="maxInclusiveValue">Maximum allowed inclusive value (optional)</param>
        public static void ConstrainInPlace(this ITensorSegment segment, float? minInclusiveValue, float? maxInclusiveValue)
        {
            MutateInPlaceParallel(segment, value => {
                if (minInclusiveValue.HasValue && value.CompareTo(minInclusiveValue.Value) < 0)
                    return minInclusiveValue.Value;
                if (maxInclusiveValue.HasValue && value.CompareTo(maxInclusiveValue.Value) > 0)
                    return maxInclusiveValue.Value;
                return value;
            });
        }

        /// <summary>
        /// Finds the average value in this tensor segment
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <returns></returns>
        public static float Average(this ITensorSegment segment) => Sum(segment) / segment.Size;

        /// <summary>
        /// Calculates the L1 norm of this tensor segment
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <returns></returns>
        public static float L1Norm(this ITensorSegment segment)
        {
            var abs = Abs(segment);
            try {
                return Sum(abs);
            }
            finally {
                abs.Release();
            }
        }

        /// <summary>
        /// Calculates the L2 norm of this tensor segment
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <returns></returns>
        public static float L2Norm(this ITensorSegment segment)
        {
            var squared = Squared(segment);
            try {
                return FloatMath.Sqrt(Sum(squared));
            }
            finally {
                squared.Release();
            }
        }

        /// <summary>
        /// Finds the min and max values (and their indices) of this tensor segment
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <returns></returns>
        public static (float Min, float Max, uint MinIndex, uint MaxIndex) GetMinAndMaxValues(this ITensorSegment segment)
        {
            var min = float.MaxValue;
            var max = float.MinValue;
            var minIndex = uint.MaxValue;
            var maxIndex = uint.MaxValue;
            uint index = 0;

            foreach (var value in segment.Values) {
                if (value.CompareTo(max) > 0) {
                    max = value;
                    maxIndex = index;
                }

                if (value.CompareTo(min) < 0) {
                    min = value;
                    minIndex = index;
                }

                ++index;
            }

            return (min, max, minIndex, maxIndex);
        }

        /// <summary>
        /// Checks if this tensor segment is finite for each value (not NaN or Infinity)
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEntirelyFinite(this ITensorSegment segment) => !segment.Values.Any(v => float.IsNaN(v) || float.IsInfinity(v));

        /// <summary>
        /// Creates a new tensor segment that is the reverse of this tensor segment
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <returns></returns>
        public static ITensorSegment Reverse(this ITensorSegment segment)
        {
            var len = segment.Size - 1;
            return TransformParallelIndexed(segment, i => segment[len - i]);
        }

        /// <summary>
        /// Splits this tensor segment into multiple contiguous tensor segments
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="blockCount">Number of blocks</param>
        /// <returns></returns>
        public static IEnumerable<ITensorSegment> Split(this ITensorSegment segment, uint blockCount)
        {
            for (uint i = 0, size = segment.Size, blockSize = size / blockCount; i < size; i += blockSize)
                yield return new TensorSegmentWrapper(segment, i, 1, blockSize);
        }

        /// <summary>
        /// Calculates the cosine distance between this and another tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <returns></returns>
        public static float CosineDistance(this ITensorSegment tensor, ITensorSegment other)
        {
            var ab = DotProduct(tensor, other);
            var aa = DotProduct(tensor, tensor);
            var bb = DotProduct(other, other);
            return 1f - ab / (FloatMath.Sqrt(aa) * FloatMath.Sqrt(bb));
        }

        /// <summary>
        /// Calculates the euclidean distance between this and another tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <returns></returns>
        public static float EuclideanDistance(this ITensorSegment tensor, ITensorSegment other)
        {
            var distance = Subtract(tensor, other);
            try {
                var squared = Squared(distance);
                try {
                    return FloatMath.Sqrt(Sum(squared));
                }
                finally {
                    squared.Release();
                }
            }
            finally {
                distance.Release();
            }
        }

        /// <summary>
        /// Calculates the mean squared distance between this and another tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <returns></returns>
        public static float MeanSquaredDistance(this ITensorSegment tensor, ITensorSegment other)
        {
            var diff = Subtract(tensor, other);
            try {
                var num = L2Norm(diff);
                return num * num / diff.Size;
            }
            finally {
                diff.Release();
            }
        }

        /// <summary>
        /// Calculates the squared euclidean distance between this and another tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <returns></returns>
        public static float SquaredEuclideanDistance(this ITensorSegment tensor, ITensorSegment other)
        {
            var diff = Subtract(tensor, other);
            try {
                var num = L2Norm(diff);
                return num * num;
            }
            finally {
                diff.Release();
            }
        }

        /// <summary>
        /// Calculates the manhattan distance between this and another tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="other">Other tensor</param>
        /// <returns></returns>
        public static float ManhattanDistance(this ITensorSegment tensor, ITensorSegment other)
        {
            var distance = Subtract(tensor, other);
            try {
                var squared = Abs(distance);
                try {
                    return Sum(squared);
                }
                finally {
                    squared.Release();
                }
            }
            finally {
                distance.Release();
            }
        }

        /// <summary>
        /// Creates a new tensor segment that contains the absolute value of each value in this tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Abs(this ITensorSegment tensor) => TransformVectorized(tensor,
            (in Vector<float> a, out Vector<float> r) => r = Vector.Abs(a),
            MathF.Abs
        );

        /// <summary>
        /// Creates a new tensor segment that contains the natural logarithm of each value in this tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Log(this ITensorSegment tensor) => TransformParallel(tensor, MathF.Log);

        /// <summary>
        /// Creates a new tensor segment that contains the exponent of each value in this tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Exp(this ITensorSegment tensor) => TransformParallel(tensor, MathF.Exp);

        /// <summary>
        /// Creates a new tensor segment that contains each value raised by the specified power in this tensor segment
        /// </summary>
        /// <param name="segment">This tensor</param>
        /// <param name="power">Specified power</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Pow(this ITensorSegment segment, float power) => TransformParallel(segment, v => FloatMath.Pow(v, power));

        /// <summary>
        /// Creates a new tensor segment that contains each value squared in this tensor segment
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Squared(this ITensorSegment tensor) => TransformVectorized(
            tensor,
            (in Vector<float> a, out Vector<float> r) => r = a * a,
            a => a * a
        );

        /// <summary>
        /// Calculates the standard deviation of this tensor segment
        /// </summary>
        /// <param name="segment">This tensor segment</param>
        /// <param name="mean">Mean of the tensor segment (optional)</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float StdDev(this ITensorSegment segment, float? mean)
        {
            var avg = mean ?? Average(segment);
            var avgVector = new Vector<float>(avg);
            var result = TransformVectorized(
                segment,
                (in Vector<float> a, out Vector<float> r) => {
                    var s = a - avgVector;
                    r = s * s;
                }, a => {
                    var s = a - avg;
                    return s * s;
                }
            );
            try {
                return FloatMath.Sqrt(Average(result));
            }
            finally {
                result.Release();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float Sigmoid(float val) => 1.0f / (1.0f + MathF.Exp(-1.0f * val));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float SigmoidDerivative(float val)
        {
            var sigmoid = Sigmoid(val);
            return sigmoid * (1.0f - sigmoid);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float Tanh(float val) => MathF.Tanh(val);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float TanhDerivative(float val) => 1.0f - MathF.Pow(Tanh(val), 2);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float Relu(float val) => (val <= 0) ? 0 : val;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float ReluDerivative(float val) => (val <= 0) ? 0f : 1;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float LeakyRelu(float val) => (val <= 0) ? 0.01f * val : val;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float LeakyReluDerivative(float val) => (val <= 0) ? 0.01f : 1;

        /// <summary>
        /// Creates a new tensor segment with sigmoid function applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment  Sigmoid(this ITensorSegment segment) => TransformParallel(segment, Sigmoid);

        /// <summary>
        /// Creates a new tensor segment with sigmoid derivative applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment SigmoidDerivative(this ITensorSegment segment) => TransformParallel(segment, SigmoidDerivative);

        /// <summary>
        /// Creates a new tensor segment with tanh function applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Tanh(this ITensorSegment segment) => TransformParallel(segment, Tanh);

        /// <summary>
        /// Creates a new tensor segment with tanh derivative applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment TanhDerivative(this ITensorSegment segment) => TransformParallel(segment, TanhDerivative);

        /// <summary>
        /// Creates a new tensor segment with RELU function applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Relu(this ITensorSegment segment) => TransformParallel(segment, Relu);

        /// <summary>
        /// Creates a new tensor segment with RELU derivative applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment ReluDerivative(this ITensorSegment segment) => TransformParallel(segment, ReluDerivative);

        /// <summary>
        /// Creates a new tensor segment with Leaky RELU function applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment LeakyRelu(this ITensorSegment segment) => TransformParallel(segment, LeakyRelu);

        /// <summary>
        /// Creates a new tensor segment with Leaky RELU derivative applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment LeakyReluDerivative(this ITensorSegment segment) => TransformParallel(segment, LeakyReluDerivative);

        /// <summary>
        /// Creates a new tensor segment with softmax function applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ITensorSegment Softmax(this ITensorSegment segment)
        {
            var (_, max, _, _) = GetMinAndMaxValues(segment);
            var softmax = segment.TransformParallel(v => MathF.Exp(v - max));
            var sum = Sum(softmax);
            if (FloatMath.IsNotZero(sum))
                softmax.MultiplyInPlace(1f / sum);
            return softmax;
        }

        /// <summary>
        /// Creates a new tensor segment with softmax derivative applied to each value in this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="lap"></param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IMatrix SoftmaxDerivative(this ITensorSegment segment, LinearAlgebraProvider lap)
        {
            return lap.CreateMatrix(segment.Size, segment.Size, (x, y) => {
                var xVal = segment[x];
                return x == y
                    ? xVal * (1 - xVal)
                    : -xVal * segment[y];
            });
        }

        /// <summary>
        /// Returns a new tensor segment from the values at the supplied indices from this tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="indices">Indices to copy to new tensor segment</param>
        /// <returns></returns>
        public static ITensorSegment CherryPickIndices(this ITensorSegment segment, params uint[] indices)
        {
            var ret = MemoryOwner<float>.Allocate(indices.Length);
            var ptr = ret.Span;
            for (int i = 0, len = indices.Length; i < len; i++)
                ptr[i] = segment[indices[i]];
            return new ArrayPoolTensorSegment(ret);
        }

        /// <summary>
        /// Rounds each value in this tensor segment to be either the lower or upper supplied parameters
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="lower"></param>
        /// <param name="upper"></param>
        public static void RoundInPlace(this ITensorSegment segment, float lower, float upper)
        {
            var compareTo = lower + (upper - lower) / 2;
            MutateInPlaceParallel(segment, v => v >= compareTo ? upper : lower);
        }

        /// <summary>
        /// Invokes a callback on each element of the tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="analyser">Callback that will receive each value and its corresponding index in the segment</param>
        public static void Analyze(ITensorSegment segment, Action<float /* value */, uint /* index */> analyser)
        {
            var size = segment.Size;
            if (size >= Consts.MinimumSizeForParallel)
                Parallel.For(0, size, i => analyser(segment[i], (uint)i));
            else {
                for (uint i = 0; i < size; i++)
                    analyser(segment[i], i);
            }
        }

        /// <summary>
        /// In place L1 regularization of the tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="coefficient">Coefficient to apply to each adjusted value</param>
        public static void L1Regularization(this ITensorSegment segment, float coefficient)
        {
            for (uint i = 0, len = segment.Size; i < len; i++) {
                var val = segment[i];
                segment[i] = val - (val > 0 ? 1 : val < 0 ? -1 : 0) * coefficient;
            }
        }

        /// <summary>
        /// Applies a distance metric to two vectors and returns the distance between them
        /// </summary>
        /// <param name="vector">First vector</param>
        /// <param name="other">Second vector</param>
        /// <param name="distance">Distance metric</param>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public static float FindDistance(this IVector vector, IVector other, DistanceMetric distance) => distance switch {
            DistanceMetric.Cosine => vector.CosineDistance(other),
            DistanceMetric.Euclidean => vector.EuclideanDistance(other),
            DistanceMetric.Manhattan => vector.ManhattanDistance(other),
            DistanceMetric.MeanSquared => vector.MeanSquaredDistance(other),
            DistanceMetric.SquaredEuclidean => vector.SquaredEuclideanDistance(other),
            _ => throw new NotImplementedException(distance.ToString())
        };

        /// <summary>
        /// Applies a distance metric to this and a list of other vectors
        /// </summary>
        /// <param name="compareTo">This vector</param>
        /// <param name="vectors">List of other vectors</param>
        /// <param name="distanceMetric">Distance metric</param>
        /// <returns>A vector in which each value is the distance between this and the corresponding other vector</returns>
        public static IVector FindDistances(this IVector compareTo, IReadOnlyList<IVector> vectors, DistanceMetric distanceMetric)
        {
            var size = (uint)vectors.Count;
            var lap = compareTo.LinearAlgebraProvider;
            var ret = lap.CreateVector(size, false);
            if (size >= Consts.MinimumSizeForParallel) {
                Parallel.For(0, size, ind => {
                    lap.BindThread();
                    ret[ind] = FindDistance(compareTo, vectors[(int)ind], distanceMetric);
                });
            }
            else {
                for (var i = 0; i < size; i++)
                    ret[i] = FindDistance(compareTo, vectors[i], distanceMetric);
            }

            return ret;
        }

        /// <summary>
        /// Sets all values of the tensor segment via a callback that receives the index of each value
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="getValue"></param>
        public static void Set(this ITensorSegment segment, Func<uint /* index */, float> getValue)
        {
            for (uint i = 0, len = segment.Size; i < len; i++)
                segment[i] = getValue(i);
        }

        /// <summary>
        /// Sets all values of the tensor segment to a single value
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="value">Value to set</param>
        public static void Set(this ITensorSegment segment, float value)
        {
            var size = segment.Size;
            if (size >= Consts.MinimumSizeForParallel) {
                Parallel.For(0, size, ind => segment[ind] = value);
            }
            else {
                for (uint i = 0, len = segment.Size; i < len; i++)
                    segment[i] = value;
            }
        }

        /// <summary>
        /// Sets all values of this tensor segment to a random floating point number
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="random">Random number generator</param>
        public static void SetToRandom(this ITensorSegment segment, Random random)
        {
            for (uint i = 0, len = segment.Size; i < len; i++)
                segment[i] = System.Convert.ToSingle(random.NextDouble());
        }

        /// <summary>
        /// Converts the vector a weighted index list (sparse vector)
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static WeightedIndexList ToSparse(this IVector vector)
        {
            return WeightedIndexList.Create(vector.Segment.Values
                .Select((v, i) => new WeightedIndexList.Item((uint)i, v))
                .Where(d => FloatMath.IsNotZero(d.Weight))
            );
        }

        /// <summary>
        /// Converts the tensor segment to a read only vector
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        public static IReadOnlyVector ToReadOnlyVector(this ITensorSegment segment) => new ReadOnlyVector(segment);

        /// <summary>
        /// Converts the float array to a read only vector
        /// </summary>
        /// <param name="segment"></param>
        /// <returns></returns>
        public static IReadOnlyVector ToReadOnlyVector(this float[] segment) => new ReadOnlyVector(segment);

        /// <summary>
        /// Creates a vector from a tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="lap">Linear algebra provider</param>
        /// <returns></returns>
        public static IVector ToVector(this ITensorSegment segment, LinearAlgebraProvider lap) => lap.CreateVector(segment);

        /// <summary>
        /// Creates a matrix from a tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="lap">Linear algebra provider</param>
        /// <param name="rows">Number of rows in matrix</param>
        /// <param name="columns">Number of columns in matrix</param>
        /// <returns></returns>
        public static IMatrix ToMatrix(this ITensorSegment segment, LinearAlgebraProvider lap, uint rows, uint columns) => lap.CreateMatrix(rows, columns, segment);

        /// <summary>
        /// Creates a 3D tensor from a tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="lap">Linear algebra provider</param>
        /// <param name="depth">Number of matrices</param>
        /// <param name="rows">Number of rows in each matrix</param>
        /// <param name="columns">Number of columns in each matrix</param>
        /// <returns></returns>
        public static ITensor3D ToTensor3D(this ITensorSegment segment, LinearAlgebraProvider lap, uint depth, uint rows, uint columns) => lap.CreateTensor3D(depth, rows, columns, segment);

        /// <summary>
        /// Creates a 4D tensor from a tensor segment
        /// </summary>
        /// <param name="segment"></param>
        /// <param name="lap">Linear algebra provider</param>
        /// <param name="count">Number of 3D tensors</param>
        /// <param name="depth">Number of matrices in each 3D tensor</param>
        /// <param name="rows">Number of rows in each matrix</param>
        /// <param name="columns">Number of columns in each matrix</param>
        /// <returns></returns>
        public static ITensor4D ToTensor4D(this ITensorSegment segment, LinearAlgebraProvider lap, uint count, uint depth, uint rows, uint columns) => lap.CreateTensor4D(count, depth, rows, columns, segment);

        /// <summary>
        /// Copies all values from this tensor to another tensor
        /// </summary>
        /// <param name="tensor">This tensor</param>
        /// <param name="other">Other tensor</param>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void CopyTo(this ITensor tensor, ITensor other) => tensor.Segment.CopyTo(other.Segment);

        /// <summary>
        /// Sets the context to use the default linear algebra provider
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        public static LinearAlgebraProvider UseDefaultLinearAlgebraProvider(this BrightDataContext context)
        {
            var ret = new LinearAlgebraProvider(context);
            ((ISetLinearAlgebraProvider)context).LinearAlgebraProvider = ret;
            return ret;
        }

        readonly struct ZipAction : IAction
        {
            readonly ITensorSegment _segment;
            readonly ITensorSegment _other;
            readonly Func<float, float, float> _action;
            readonly float[] _ret;

            public ZipAction(ITensorSegment segment, ITensorSegment other, Func<float, float, float> action, float[] ret)
            {
                _segment = segment;
                _other = other;
                _action = action;
                _ret = ret;
            }

            public void Invoke(int i) => _ret[i] = _action(_segment[i], _other[i]);
        }
        readonly struct TransformAction : IAction
        {
            readonly ITensorSegment _segment;
            readonly Func<float, float> _action;
            readonly float[] _ret;

            public TransformAction(ITensorSegment segment, Func<float, float> action, float[] ret)
            {
                _segment = segment;
                _action = action;
                _ret = ret;
            }

            public void Invoke(int i) => _ret[i] = _action(_segment[i]);
        }
        readonly struct TransformIndexedAction : IAction
        {
            readonly Func<uint, float> _action;
            readonly float[] _ret;

            public TransformIndexedAction(Func<uint, float> action, float[] ret)
            {
                _action = action;
                _ret = ret;
            }

            public void Invoke(int i) => _ret[i] = _action((uint)i);
        }
        readonly struct MutateAction : IAction
        {
            readonly Func<float, float> _action;
            readonly ITensorSegment _segment;

            public MutateAction(Func<float, float> action, ITensorSegment segment)
            {
                _action = action;
                _segment = segment;
            }

            public void Invoke(int i) => _segment[i] = _action(_segment[i]);
        }
        
        internal static uint[] ResolveShape(this uint total, params uint?[] shape)
        {
            uint nonNullTotal = 1;
            var hasFoundNull = false;
            foreach (var item in shape) {
                if (item.HasValue)
                    nonNullTotal *= item.Value;
                else if (!hasFoundNull)
                    hasFoundNull = true;
                else
                    throw new ArgumentException("Only one parameter can be null");
            }

            if (hasFoundNull && nonNullTotal == 0)
                throw new ArgumentException("Cannot resolve null parameter");

            if (!hasFoundNull && nonNullTotal != total)
                throw new ArgumentException($"Invalid shape arguments: {String.Join("x", shape)} == {nonNullTotal:N0} but expected to be {total:N0}");

            return shape.Select(v => v ?? total / nonNullTotal).ToArray();
        }

        /// <summary>
        /// Calculates the pearson correlation coefficient metric between two tensor segments
        /// https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="DivideByZeroException"></exception>
        public static double? PearsonCorrelationCoefficient(this ITensorSegment x, ITensorSegment y)
        {
            var size = x.Size;
            if (size != y.Size)
                throw new ArgumentException("The segments must have the same length.");

            // calculate the means of x and y
            var xMean = x.Average();
            var yMean = y.Average();

            // calculate the numerator and denominator of the Pearson similarity formula
            var numerator = 0f;
            var denominatorX = 0f;
            var denominatorY = 0f;

            for (var i = 0; i < size; i++) {
                // get the deviations from the means
                var xDeviation = x[i] - xMean;
                var yDeviation = y[i] - yMean;

                // update the numerator and denominator
                numerator += xDeviation * yDeviation;
                denominatorX += xDeviation * xDeviation;
                denominatorY += yDeviation * yDeviation;
            }

            // calculate the Pearson similarity
            var denominator = MathF.Sqrt(denominatorX * denominatorY);
    
            // check if the denominator is zero
            if (denominator == 0)
                throw new DivideByZeroException("The standard deviations of x and y are zero.");

            return numerator / denominator;
        }
    }
}
