using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;

namespace BrightData.LinearAlgebra.VectorIndexing.IndexStrategy
{
    internal class FlatVectorIndex<T>(IStoreVectors<T> storage, DistanceMetric distanceMetric) : IVectorIndex<T>
        where T : unmanaged, IBinaryFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        public IStoreVectors<T> Storage { get; } = storage;
        public uint Add(ReadOnlySpan<T> vector) => Storage.Add(vector);

        public void Dispose()
        {
            Storage.Dispose();
        }

        public IEnumerable<uint> Rank(ReadOnlySpan<T> vector)
        {
            // ReadOnlySpan<T> is a ref struct and cannot be used in a yield-return method.
            // Compute results eagerly and return the array directly.
            return ComputeRankedIndices(vector);
        }

        // Compute distances, sort by distance, and return sorted indices (skipping NaN).
        uint[] ComputeRankedIndices(ReadOnlySpan<T> vector)
        {
            var size = Storage.Size;
            var distances = new T[size];
            var indices = new uint[size];

            // Copy to array so it can be captured by the lambda callback.
            var vectorArray = vector.ToArray();

            for (var i = 0U; i < size; i++)
                indices[i] = i;

            Storage.ForEach((x, i) => distances[i] = x.FindDistance(vectorArray, distanceMetric));
            Array.Sort(distances, indices, 0, (int)size);

            // Filter out NaN distances and compact into result array.
            var result = new List<uint>((int)size);
            for (var i = 0U; i < size; i++)
            {
                if (!T.IsNaN(distances[i]))
                    result.Add(indices[i]);
            }
            return result.ToArray();
        }

        public uint[] Closest(ReadOnlyMemory<T>[] vector)
        {
            var size = Storage.Size;
            var distance = GetDistance(vector);

            // find the closest input vector index for each vector in the set
            var ret = new uint[size];
            Parallel.For(0, size, i => ret[i] = ((ReadOnlySpan<T>)distance.AsSpan2D().GetRowSpan((int)i)).MinimumIndex());
            return ret;
        }

        /// <summary>
        /// Find distance between each vector in the set and each input vector
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        T[,] GetDistance(ReadOnlyMemory<T>[] vector)
        {
            var size = Storage.Size;
            var ret = new T[size, vector.Length];
            Parallel.For(0, size * vector.Length, i =>
            {
                var dataIndex = (uint)i % size;
                var vectorIndex = (uint)i / size;
                ret[dataIndex, vectorIndex] = Storage[dataIndex].FindDistance(vector[vectorIndex].Span, distanceMetric);
            });
            return ret;
        }
    }
}
