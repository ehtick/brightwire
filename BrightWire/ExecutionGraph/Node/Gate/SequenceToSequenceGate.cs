﻿using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using BrightData;
using BrightWire.ExecutionGraph.Helper;

namespace BrightWire.ExecutionGraph.Node.Gate
{
    class SequenceToSequenceGate : NodeBase
    {
        ConcurrentStack<IGraphSequenceContext> _encoderContext;

        public SequenceToSequenceGate(string? name, string? id = null) : base(name, id)
        {
        }

        public override (NodeBase FromNode, IGraphData Output, Func<IBackpropagate>? BackProp) ForwardInternal(IGraphData signal, uint channel, IGraphSequenceContext context, NodeBase? source)
        {
            _encoderContext ??= new ConcurrentStack<IGraphSequenceContext>();
            _encoderContext.Push(context);
            if (context.BatchSequence.Type == MiniBatchSequenceType.SequenceEnd) {
                var nextBatch = context.BatchSequence.MiniBatch.NextMiniBatch;
                if (nextBatch == null)
                    throw new Exception("No following mini batch was found");

                context.ExecutionContext.RegisterAdditional(nextBatch, signal, OnStartEncoder, OnEndEncoder);
            }

            return (this, GraphData.Null, null);
        }

        void OnStartEncoder(IGraphSequenceContext context, IGraphData data)
        {
            foreach (var wire in Output)
                wire.SendTo.Forward(data, context, wire.Channel);

            //AddNextGraphAction(context, data, null/*, () => new Backpropagation(this)*/);
        }

        void OnEndEncoder(IGraphSequenceContext[] context)
        {
            var learningContext = context.FirstOrDefault()?.LearningContext;
            if (learningContext != null) {
                var gradient = learningContext.BackpropagateThroughTime(null);

                //var firstContext = (ICanTrace)context.Single(c => c.BatchSequence.Type == MiniBatchSequenceType.SequenceStart);
                //var lastContext = (ICanTrace)context.Single(c => c.BatchSequence.Type == MiniBatchSequenceType.SequenceEnd);

                //firstContext.Trace();
                //lastContext.Trace();
                
                if (gradient != null) {
                    foreach (var item in _encoderContext.Reverse())
                        learningContext.DeferBackpropagation(null, delta => item.Backpropagate(delta));
                    learningContext.BackpropagateThroughTime(gradient);
                }
            }

            _encoderContext.Clear();
        }
    }
}