﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using CommunityToolkit.HighPerformance.Buffers;

namespace BrightData.Types.Graph
{
    public class DirectedGraphBuilder<T>
        where T: struct, IHaveSingleIndex
    {
        class Node(T value, int nodeOffset)
        {
            HashSet<int>? _edges = null;
            public T Value { get; } = value;
            public int NodeOffset { get; } = nodeOffset;

            public void AddEdge(int toNodeOffset)
            {
                (_edges??=new()).Add(toNodeOffset);
            }
            public IEnumerable<int> ConnectedToNodeOffsets => _edges ?? Enumerable.Empty<int>();
            public int NumEdges => _edges?.Count ?? 0;
        }
        readonly Dictionary<uint /* node index */, Node> _nodes = new();

        public uint Add(T node)
        {
            var nodeIndex = node.Index;
            _nodes.Add(nodeIndex, new Node(node, _nodes.Count));
            return nodeIndex;
        }

        public bool AddEdge(uint fromNodeIndex, uint toNodeIndex)
        {
            if (_nodes.TryGetValue(fromNodeIndex, out var from) && _nodes.TryGetValue(toNodeIndex, out var to)) {
                from.AddEdge(to.NodeOffset);
                return true;
            }

            return false;
        }

        public DirectedGraph<T> Build()
        {
            using var nodes = new ArrayPoolBufferWriter<DirectedGraph<T>.Node>();
            using var edges = new ArrayPoolBufferWriter<int>();

            foreach (var node in _nodes.OrderBy(x => x.Key)) {
                var edgeIndex = edges.WrittenCount;
                var edgeCount = node.Value.NumEdges;
                if (edgeCount > 0) {
                    var span = edges.GetSpan((int)edgeCount);
                    var index = 0;
                    foreach (var edge in node.Value.ConnectedToNodeOffsets.Order())
                        span[index++] = edge;
                    edges.Advance((int)edgeCount);
                }
                nodes.Write(new DirectedGraph<T>.Node(node.Value.Value, edgeIndex, edgeCount));
            }

            return new DirectedGraph<T>(nodes.WrittenSpan, edges.WrittenSpan);
        }
    }
}
