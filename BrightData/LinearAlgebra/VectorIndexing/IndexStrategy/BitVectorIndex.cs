using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using BrightData.Types;
using CommunityToolkit.HighPerformance.Buffers;

namespace BrightData.LinearAlgebra.VectorIndexing.IndexStrategy
{
    /// <summary>
    /// Vector index that uses bit vectors for fast Hamming distance computation.
    /// Each vector is converted to a binary signature where a bit is set if the
    /// corresponding dimension is positive.
    /// </summary>
    /// <typeparam name="T">The numeric type of the vector elements.</typeparam>
    internal class BitVectorIndex<T>(IStoreVectors<T> storage) : IVectorIndex<T>
        where T : unmanaged, IBinaryFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        readonly ArrayPoolBufferWriter<ulong> _indexStorage = new();
        readonly uint _bitVectorWords = BitVector.GetRequiredSize(storage.VectorSize);

        /// <inheritdoc />
        public IStoreVectors<T> Storage { get; } = storage;

        /// <inheritdoc />
        public void Dispose()
        {
            Storage.Dispose();
            _indexStorage.Dispose();
        }

        /// <inheritdoc />
        public uint Add(ReadOnlySpan<T> vector)
        {
            if ((uint)vector.Length != Storage.VectorSize)
                throw new ArgumentException($"Vector length must match storage.VectorSize ({Storage.VectorSize}).", nameof(vector));

            // Commit to storage first so index never gets ahead of storage.
            var index = Storage.Add(vector);
            _indexStorage.Write(AsBits(vector).AsSpan());
            return index;
        }

        /// <summary>
        /// Converts a dense vector into a bit vector where each bit indicates
        /// whether the corresponding dimension is positive.
        /// </summary>
        /// <param name="vector">The source vector.</param>
        /// <returns>A bit vector representation.</returns>
        static BitVector AsBits(ReadOnlySpan<T> vector)
        {
            var index = 0;
            var ret = new BitVector((uint)vector.Length);
            foreach (var value in vector)
            {
                if (value > T.Zero)
                    ret[index] = true;
                ++index;
            }
            return ret;
        }

        /// <summary>
        /// Computes the Hamming distance between two bit vectors represented as
        /// spans of <see cref="ulong"/> words, without any allocation.
        /// </summary>
        /// <param name="a">First bit vector data.</param>
        /// <param name="b">Second bit vector data.</param>
        /// <returns>Number of differing bits.</returns>
        static unsafe uint HammingDistance(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
        {
            var count = 0;
            fixed (ulong* pa = a)
            fixed (ulong* pb = b)
            {
                var x = pa;
                var y = pb;
                for (var i = 0; i < a.Length; i++)
                    count += BitOperations.PopCount(*x++ ^ *y++);
            }
            return (uint)count;
        }

        /// <inheritdoc />
        public IEnumerable<uint> Rank(ReadOnlySpan<T> vector)
        {
            if ((uint)vector.Length != Storage.VectorSize)
                throw new ArgumentException($"Vector length must match storage.VectorSize ({Storage.VectorSize}).", nameof(vector));

            var len = Storage.Size;
            var results = new uint[len];
            var span = _indexStorage.WrittenSpan;
            var comparison = AsBits(vector);

            for (var i = 0U; i < len; i++)
            {
                var existingSpan = span.Slice((int)(i * _bitVectorWords), (int)_bitVectorWords);
                results[i] = HammingDistance(existingSpan, comparison.AsSpan());
            }

            return results
                .Select((x, i) => (Distance: x, Index: (uint)i))
                .OrderBy(x => x.Distance)
                .Select(x => x.Index)
            ;
        }

        /// <inheritdoc />
        public uint[] Closest(ReadOnlyMemory<T>[] vectors)
        {
            if (vectors.Length == 0)
                return Array.Empty<uint>();

            var len = Storage.Size;
            if (len == 0)
                throw new InvalidOperationException("Index is empty.");

            var results = new uint[vectors.Length];
            var span = _indexStorage.WrittenSpan;

            // Reusable buffer for the query bit vector to avoid per-query allocations.
            var bits = new ulong[_bitVectorWords];

            for (var idx = 0; idx < vectors.Length; idx++)
            {
                // Convert query vector to bit representation.
                AsBitsInto(vectors[idx].Span, bits);

                var bestIndex = 0U;
                var bestDistance = uint.MaxValue;

                for (var i = 0U; i < len; i++)
                {
                    var existingSpan = span.Slice((int)(i * _bitVectorWords), (int)_bitVectorWords);
                    var distance = HammingDistance(existingSpan, bits);

                    if (distance < bestDistance)
                    {
                        bestDistance = (uint)distance;
                        bestIndex = i;
                    }
                }

                results[idx] = bestIndex;
            }

            return results;
        }

        /// <summary>
        /// Converts a dense vector into a bit vector, writing the result directly
        /// into the provided buffer to avoid allocations.
        /// </summary>
        /// <param name="vector">The source vector.</param>
        /// <param name="bits">Buffer to receive the bit vector words.</param>
        static void AsBitsInto(ReadOnlySpan<T> vector, Span<ulong> bits)
        {
            bits.Clear();
            for (var i = 0; i < vector.Length; i++)
            {
                if (vector[i] > T.Zero)
                {
                    var wordIndex = i / BitVector.NumBitsPerItem;
                    var bitOffset = i % BitVector.NumBitsPerItem;
                    bits[wordIndex] |= 1UL << bitOffset;
                }
            }
        }
    }
}
