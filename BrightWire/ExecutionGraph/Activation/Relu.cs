﻿using BrightWire.ExecutionGraph.Node;
using BrightData;

namespace BrightWire.ExecutionGraph.Activation
{
    /// <summary>
    /// RELu activation
    /// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    /// </summary>
    internal class Relu : NodeBase
    {
        class Backpropagation : SingleBackpropagationBase<Relu>
        {
            readonly IFloatMatrix _input;

            public Backpropagation(Relu source, IFloatMatrix matrix) : base(source)
            {
                _input = matrix;
            }

            protected override IGraphData Backpropagate(INode? fromNode, IGraphData errorSignal, IGraphSequenceContext context, INode[] parents)
            {
                using var od = _input.ReluDerivative();
                var delta = errorSignal.GetMatrix().PointwiseMultiply(od);
                return errorSignal.ReplaceWith(delta);
            }
        }

        public Relu(string? name = null) : base(name) { }

        public override void ExecuteForward(IGraphSequenceContext context)
        {
            var input = context.Data.GetMatrix();
            var output = context.Data.ReplaceWith(input.ReluActivation());
            AddNextGraphAction(context, output, () => new Backpropagation(this, input));
        }
    }
}
