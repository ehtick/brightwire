﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using BrightData.Types;
using CommunityToolkit.HighPerformance;

namespace BrightData.LinearAlgebra.VectorIndexing.Helper
{
    /// <summary>
    /// Creates a graph of vectors with a fixed size set of neighbours
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class VectorGraph<T> : IHaveSize
        where T : unmanaged, IBinaryFloatingPointIeee754<T>, IMinMaxValue<T>
    {
        readonly IndexedFixedSizeGraphNode<T, FixedSizeSortedAscending8Array<uint, T>>[] _nodes;

        /// <summary>
        /// Creates a vector graph from an array of nodes
        /// </summary>
        /// <param name="nodes"></param>
        public VectorGraph(IndexedFixedSizeGraphNode<T, FixedSizeSortedAscending8Array<uint, T>>[] nodes)
        {
            _nodes = nodes;
        }

        /// <summary>
        /// Number of nodes in the graph
        /// </summary>
        public uint Size => (uint)_nodes.Length;

        /// <summary>
        /// Gets the neighbours for a node, sorted by distance
        /// </summary>
        /// <param name="vectorIndex"></param>
        /// <returns></returns>
        public ReadOnlySpan<uint> GetNeighbours(uint vectorIndex) => _nodes[vectorIndex].NeighbourIndices;

        /// <summary>
        /// Gets the weights for the node's neighbours
        /// </summary>
        /// <param name="vectorIndex"></param>
        /// <returns></returns>
        public ReadOnlySpan<T> GetNeighbourWeights(uint vectorIndex) => _nodes[vectorIndex].NeighbourWeights;

        /// <summary>
        /// Enumerates the neighbour indices and their weights in ascending order
        /// </summary>
        /// <param name="vectorIndex"></param>
        /// <returns></returns>
        public IEnumerable<(uint NeighbourIndex, T NeighbourWeight)> GetWeightedNeighbours(uint vectorIndex) => _nodes[vectorIndex].WeightedNeighbours;

        /// <summary>
        /// Creates 
        /// </summary>
        /// <param name="vectors"></param>
        /// <param name="distanceMetric"></param>
        /// <param name="shortCircuitIfNodeNeighboursAreFull"></param>
        /// <param name="onNode"></param>
        /// <returns></returns>
        [SkipLocalsInit]
        public static unsafe VectorGraph<T> Build(
            IStoreVectors<T> vectors, 
            DistanceMetric distanceMetric, 
            bool shortCircuitIfNodeNeighboursAreFull = true,
            Action<uint>? onNode = null)
        {
            var size = vectors.Size;
            var distance = size <= 1024 
                ? stackalloc T[(int)size] 
                : new T[size];

            var ret = GC.AllocateUninitializedArray<IndexedFixedSizeGraphNode<T, FixedSizeSortedAscending8Array<uint, T>>>((int)size);
            for (var i = 0U; i < size; i++)
                ret[i] = new(i);

            for (var i = 0U; i < size; i++)
            {
                if (shortCircuitIfNodeNeighboursAreFull && ret[i].NeighbourCount == FixedSizeSortedAscending8Array<uint, T>.MaxSize)
                    continue;

                // find the distance between this node and each of its neighbours
                fixed (T* dest = distance)
                {
                    var destPtr = dest;
                    var currentIndex = i;
                    vectors.ForEach((x, j) =>
                    {
                        if(currentIndex != j)
                            destPtr[j] = T.Abs(x.FindDistance(vectors[currentIndex], distanceMetric));
                    });
                }

                // find top N closest neighbours
                var maxHeap = new IndexedFixedSizeGraphNode<T, FixedSizeSortedAscending8Array<uint, T>>();
                for (var j = 0; j < size; j++) {
                    if (i == j)
                        continue;
                    var d = distance[j];
                    maxHeap.TryAddNeighbour((uint)j, d);
                }

                // connect the closest nodes
                foreach (var (index, d) in maxHeap.WeightedNeighbours)
                {
                    ret[index].TryAddNeighbour(i, d);
                    ret[i].TryAddNeighbour(index, d);
                }
                onNode?.Invoke(i);
            }

            return new(ret);
        }

        /// <summary>
        /// Writes the graph to disk
        /// </summary>
        /// <param name="filePath"></param>
        public async Task WriteToDisk(string filePath)
        {
            using var fileHandle = File.OpenHandle(filePath, FileMode.Create, FileAccess.Write, FileShare.None, FileOptions.Asynchronous);
            await RandomAccess.WriteAsync(fileHandle, _nodes.AsMemory().AsBytes(), 0);
        }

        /// <summary>
        /// Loads a vector graph from disk
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static async Task<VectorGraph<T>> LoadFromDisk(string filePath)
        {
            using var fileHandle = File.OpenHandle(filePath);
            var ret = GC.AllocateUninitializedArray<IndexedFixedSizeGraphNode<T, FixedSizeSortedAscending8Array<uint, T>>>((int)(RandomAccess.GetLength(fileHandle) / Unsafe.SizeOf<IndexedFixedSizeGraphNode<T, FixedSizeSortedAscending8Array<uint, T>>>()));
            await RandomAccess.ReadAsync(fileHandle, ret.AsMemory().AsBytes(), 0);
            return new(ret);
        }
    }
}
