﻿using System.Linq;
using BrightData.Helper;
using BrightWire.Helper;

namespace BrightWire.ExecutionGraph.Action
{
    /// <summary>
    /// Backpropagates through time (for recurrent neural networks)
    /// </summary>
    internal class BackpropagateThroughTime : IAction
    {
        IErrorMetric _errorMetric;

        public BackpropagateThroughTime(IErrorMetric errorMetric)
        {
            _errorMetric = errorMetric;
        }

        public void Initialise(string data)
        {
            _errorMetric = GenericActivator.Create<IErrorMetric>(TypeLoader.LoadType(data));
        }

        public string Serialise()
        {
            return TypeLoader.GetTypeName(_errorMetric);
        }

        public IGraphData Execute(IGraphData input, IGraphSequenceContext context)
        {
            var output = input.GetMatrix();
            if (context.LearningContext != null) {
	            context.LearningContext.ErrorMetric ??= _errorMetric;

                var batchSequence = context.BatchSequence;
                var target = batchSequence.Target?.GetMatrix();
                if (target == null)
                    context.LearningContext.DeferBackpropagation(null, context.Backpropagate);
                else {
                    var gradient = _errorMetric.CalculateGradient(context, output, target);
                    context.LearningContext.DeferBackpropagation(input.ReplaceWith(gradient), context.Backpropagate);
                }
            }
            return input;
        }
    }
}
