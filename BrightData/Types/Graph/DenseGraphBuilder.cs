﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrightData.Types.Graph
{
    public class DenseGraphBuilder<T> : IBuildGraphs<T>
        where T: unmanaged
    {
        readonly record struct NodeIndexPair(uint FirstNodeIndex, uint SecondNodeIndex);

        readonly List<T> _nodes = new();
        readonly HashSet<NodeIndexPair> _edges = new();
        readonly HashSet<uint> _validIndices = new();

        /// <inheritdoc />
        public uint Add(T node)
        {
            var nodeIndex = (uint)_nodes.Count;
            _nodes.Add(node);
            _validIndices.Add(nodeIndex);
            return nodeIndex;
        }

        /// <inheritdoc />
        public bool AddEdge(uint fromNodeIndex, uint toNodeIndex)
        {
            if (_validIndices.Contains(fromNodeIndex) && _validIndices.Contains(toNodeIndex)) {
                _edges.Add(new(fromNodeIndex, toNodeIndex));
                return true;
            }

            return false;
        }

        public DenseGraph<T> Build()
        {
            var nodeCount = (uint)_nodes.Count;
            var edges = new BitVector(nodeCount * nodeCount);
            foreach (var (from, to) in _edges) {
                edges[DenseGraph<T>.GetEdgeIndex(from, to, nodeCount)] = true;
            }
            return new DenseGraph<T>(_nodes.ToArray(), edges);
        }
    }
}
