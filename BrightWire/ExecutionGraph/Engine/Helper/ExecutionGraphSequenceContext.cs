﻿using System;
using System.Collections.Generic;
using System.Linq;
using BrightData;
using BrightWire.ExecutionGraph.Helper;
using BrightWire.ExecutionGraph.Node;
using BrightWire.Models;

namespace BrightWire.ExecutionGraph.Engine.Helper
{
    /// <summary>
    /// Execution engine context
    /// </summary>
    internal class ExecutionGraphSequenceContext : IGraphSequenceContext
    {
        readonly IGraphExecutionContext _executionContext;
        readonly List<ExecutionHistory> _forward = new List<ExecutionHistory>();
	    readonly Dictionary<int, IGraphData> _output = new Dictionary<int, IGraphData>();
        NodeBase? _sourceNode;

        public ExecutionGraphSequenceContext(IGraphExecutionContext executionContext, IMiniBatchSequence miniBatch)
        {
            _executionContext = executionContext;
            BatchSequence = miniBatch;
            Data = GraphData.Null;
        }

        public void Dispose()
        {
            // nop
        }

        public bool IsTraining => false;
        public NodeBase? Source => _sourceNode;
        public IGraphExecutionContext ExecutionContext => _executionContext;
        public ILearningContext? LearningContext => null;
        public ILinearAlgebraProvider LinearAlgebraProvider => _executionContext.LinearAlgebraProvider;
        public IMiniBatchSequence BatchSequence { get; }
        public IGraphData? Backpropagate(IGraphData? delta) => throw new NotImplementedException();
        public void AddForward(NodeBase source, IGraphData data, Func<IBackpropagate>? callback, params NodeBase[] prev) => _forward.Add(new ExecutionHistory(source, data));
        public IGraphData ErrorSignal => throw new NotImplementedException();
        //public bool HasNext => _forward.Any();
        public IGraphData Data { get; set; }

        //public bool ExecuteNext()
        //{
        //    if (HasNext) {
        //        var next = _forward.ElementAt(0);
        //        _forward.RemoveAt(0);

        //        Data = next.Data;

        //        _sourceNode = next.Source;
        //        foreach (var output in next.Source.Output)
        //            output.SendTo.ExecuteForward(this, output.Channel);

        //        return true;
        //    }
        //    return false;
        //}

	    public void SetOutput(IGraphData data, int channel = 0)
	    {
		    _output[channel] = data;
	    }

	    public IGraphData? GetOutput(int channel = 0)
	    {
		    if (_output.TryGetValue(channel, out var ret))
			    return ret;
		    return null;
	    }

	    public IGraphData[] Output => _output
            .OrderBy(kv => kv.Key)
            .Select(kv => kv.Value)
            .ToArray()
        ;

        public IEnumerable<ExecutionResult> Results 
        {
            get
            {
                if (Data.HasValue) {
                    yield return  new ExecutionResult(BatchSequence, Data.GetMatrix().Data.Rows.ToArray());
                }
                //var output = Output;
                //var matrixOutput = output.Any()
                //    ? output.Select(o => o.GetMatrix().Data)
                //    : new[] {Data.GetMatrix().Data};

                //return new ExecutionResult(BatchSequence, matrixOutput.SelectMany(m => m.Rows).ToArray());
            }
        }

        public void ClearForBackpropagation()
        {
            // nop
        }
    }
}