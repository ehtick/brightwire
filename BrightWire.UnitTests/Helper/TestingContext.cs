﻿using System;
using System.Collections.Generic;
using BrightData;
using BrightWire.ExecutionGraph.Helper;
using BrightWire.Models;

namespace BrightWire.UnitTests.Helper
{
    internal class TestingContext : IGraphSequenceContext
    {
        public List<(ExecutionHistory, IBackpropagate)> Forward { get; } = new List<(ExecutionHistory, IBackpropagate)>();

        public TestingContext(ILinearAlgebraProvider lap)
        {
            LinearAlgebraProvider = lap;
            LearningContext = new MockLearningContext();
        }

        public void Dispose()
        {
            // nop
        }

        public INode Source { get; }
        public IGraphData Data { get; set; }
        public IGraphExecutionContext ExecutionContext { get; }
        public ILearningContext LearningContext { get; }
        public ILinearAlgebraProvider LinearAlgebraProvider { get; }

        public IMiniBatchSequence BatchSequence { get; }
        public void AddForward(ExecutionHistory action, Func<IBackpropagate>? callback)
        {
            Forward.Add((action, callback()));
        }

        public void AppendErrorSignal(IGraphData errorSignal, INode forNode)
        {
            throw new NotImplementedException();
        }

        public void AddForward(INode source, IGraphData data, Func<IBackpropagate>? callback, params INode[] prev)
        {
            Forward.Add((new ExecutionHistory(source, data), callback()));
        }

        public IGraphData? Backpropagate(IGraphData? delta)
        {
            throw new NotImplementedException();
        }

        public IGraphData ErrorSignal { get; }
        public bool HasNext { get; }
        public bool ExecuteNext()
        {
            throw new NotImplementedException();
        }

        public void SetOutput(IGraphData data, int channel = 0)
        {
            throw new NotImplementedException();
        }

        public IGraphData GetOutput(int channel = 0)
        {
            throw new NotImplementedException();
        }

        public IGraphData[] Output { get; set; }

        public ExecutionResult Result { get; }
    }
}
