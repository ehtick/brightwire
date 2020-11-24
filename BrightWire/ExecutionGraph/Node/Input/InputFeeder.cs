﻿using System.IO;

namespace BrightWire.ExecutionGraph.Node.Input
{
    class InputFeeder : NodeBase
    {
        uint _index;

        public InputFeeder(uint index, string name = null) : base(name)
        {
            _index = index;
        }

        public override void ExecuteForward(IGraphContext context)
        {
            var input = context.BatchSequence.Input[_index];
            _AddNextGraphAction(context, input, null);
        }

        protected override (string Description, byte[] Data) _GetInfo()
        {
            return ("INPUT", _WriteData(WriteTo));
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
