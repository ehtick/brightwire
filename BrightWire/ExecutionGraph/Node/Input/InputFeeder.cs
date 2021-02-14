﻿using System;
using System.IO;

namespace BrightWire.ExecutionGraph.Node.Input
{
    internal class InputFeeder : NodeBase
    {
        uint _index;

        public InputFeeder(uint index, string? name = null) : base(name)
        {
            _index = index;
            if (index != 0)
                throw new NotImplementedException("Only one input is now supported");
        }

        public override (NodeBase FromNode, IGraphData Output, Func<IBackpropagate>? BackProp) ForwardInternal(IGraphData signal, uint channel, IGraphSequenceContext context, NodeBase? source)
        {
            if (_index == 0) {
                var input = context.BatchSequence.Input;
                if (input == null)
                    throw new Exception("Input data was null");

                return (this, input, null);
            }
            else
                throw new NotImplementedException();
        }

        protected override (string Description, byte[] Data) GetInfo()
        {
            return ("INPUT", WriteData(WriteTo));
        }

        public override void WriteTo(BinaryWriter writer)
        {
            writer.Write((int)_index);
        }

        public override void ReadFrom(GraphFactory factory, BinaryReader reader)
        {
            _index = (uint)reader.ReadInt32();
        }
    }
}
