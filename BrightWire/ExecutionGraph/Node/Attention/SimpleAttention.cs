﻿using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BrightData;
using BrightData.LinearAlgebra;
using BrightWire.ExecutionGraph.Node.Input;
using BrightData.Helper;

namespace BrightWire.ExecutionGraph.Node.Attention
{
    internal class SimpleAttention : NodeBase
    {
        LinearAlgebraProvider<float> _lap;
        string                       _encoderName, _decoderName;
        uint                         _inputSize, _encoderSize, _decoderSize, _blockSize;
        IGradientDescentOptimisation _updater;
        IMatrix<float>               _attention;

        class Backpropagation(SimpleAttention source, uint position, uint sequenceSize, IMatrix<float> inputMatrix, INumericSegment<float>[] softmax)
            : SingleBackpropagationBase<SimpleAttention>(source)
        {
            protected override IGraphData Backpropagate(IGraphData errorSignal, IGraphContext context)
            {
                var (left, right) = errorSignal.GetMatrix().SplitAtColumn(position);
                Debug.Assert(right.IsEntirelyFinite());

                // train the attention layer
                var learningContext = context.LearningContext!;
                var lap = context.GetLinearAlgebraProvider();
                using var errorTensor = lap.CreateTensor3D(sequenceSize.AsRange().Select(_ => right).ToArray());
                using var errorMatrix = errorTensor.Reshape(inputMatrix.RowCount, inputMatrix.ColumnCount);
                using var weightError = errorMatrix.PointwiseMultiply(inputMatrix);
                using var weightVector = weightError.RowSums();
                weightVector.MultiplyInPlace(1f / weightError.ColumnCount);

                using var weightMatrix = weightVector.Reshape(null, sequenceSize);
                var softmaxError = weightMatrix.SoftmaxDerivativePerRow(softmax);
                //using var softmaxErrorMatrix = lap.CreateMatrixFromColumns(softmaxError);
                using var softmaxErrorMatrix = lap.CreateMatrixFromRows(softmaxError);
                using var softmaxErrorMatrix2 = softmaxErrorMatrix.Reshape(null, 1);
                var attentionError = softmaxErrorMatrix2.TransposeThisAndMultiply(inputMatrix);
                learningContext.AddError(NodeErrorType.Default, _source, attentionError);
                softmaxError.DisposeAll();

                return left.AsGraphData();
            }

            protected override void DisposeMemory(bool isDisposing)
            {
                if (isDisposing) {
                    softmax.DisposeAll();
                }
            }
        }

        public SimpleAttention(
            LinearAlgebraProvider<float> lap, 
            string encoderName, 
            string decoderName,
            uint inputSize, 
            uint encoderSize, 
            uint decoderSize,
            IWeightInitialisation weightInit,
            Func<IMatrix<float>, IGradientDescentOptimisation> updater,
            string? name, 
            string? id = null
        ) : base(name, id)
        {
            _lap         = lap;
            _encoderName = encoderName;
            _decoderName = decoderName;
            _inputSize   = inputSize;
            _encoderSize = encoderSize;
            _decoderSize = decoderSize;
            _attention   = weightInit.CreateWeight(1, _blockSize = _inputSize + _encoderSize + _decoderSize);
            _updater     = updater(_attention);
        }

        public override (NodeBase FromNode, IGraphData Output, Func<IBackpropagate>? BackProp) ForwardSingleStep(IGraphData signal, uint channel, IGraphContext context, NodeBase? source)
        {
            var currentIndex = context.BatchSequence.SequenceIndex;
            var batchSize = context.BatchSequence.MiniBatch.BatchSize;

            // get the previous decoder state
            IMatrix<float>? decoderHiddenState = null;
            if (_decoderSize > 0) {
                if (currentIndex == 0) {
                    if (FindByName(_decoderName) is IHaveMemoryNode { Memory: MemoryFeeder decoderMemory })
                        decoderHiddenState = context.ExecutionContext.GetMemory(decoderMemory.Id);
                }
                else {
                    var previousContext = context.BatchSequence.MiniBatch.GetSequenceAtIndex(currentIndex - 1).GraphContext;
                    decoderHiddenState = previousContext?.GetData("hidden-forward").Single(d => d.Name == _decoderName).Data.GetMatrix();
                }

                if (decoderHiddenState == null)
                    throw new Exception("Not able to find the decoder hidden state");
                if (decoderHiddenState.ColumnCount != _decoderSize)
                    throw new Exception($"Expected decoder size to be {_decoderSize} but found {decoderHiddenState.ColumnCount}");
            }

            // find each encoder hidden state and sequence input
            var previousBatch = context.BatchSequence.MiniBatch.PreviousMiniBatch ?? throw new Exception("No previous mini batch");
            Debug.Assert(batchSize == previousBatch.BatchSize);
            IMatrix<float>[]? encoderStates = null;
            var inputs = new IMatrix<float>[previousBatch.SequenceCount];
            for (uint i = 0, len = previousBatch.SequenceCount; i < len; i++) {
                var sequence = previousBatch.GetSequenceAtIndex(i);
                if (_encoderSize > 0) {
                    if (i == 0)
                        encoderStates = new IMatrix<float>[len];
                    var encoderState = sequence.GraphContext!.GetData("hidden-forward").Single(d => d.Name == _encoderName).Data.GetMatrix()
                        ?? throw new Exception("Not able to find the encoder hidden state");
                    if (encoderState.ColumnCount != _encoderSize)
                        throw new Exception($"Expected encoder size to be {_encoderSize} but found {encoderState.ColumnCount}");
                    encoderStates![i] = encoderState;
                }

                var input = sequence.Input?.GetMatrix()
                    ?? throw new Exception("Not able to find the input matrix");
                if (input.ColumnCount != _inputSize)
                    throw new Exception($"Expected input size to be {_inputSize} but found {input.ColumnCount}");
                inputs[i] = input;
            }

            var sequenceSize = previousBatch.SequenceCount;
            var numInputRows = batchSize * sequenceSize;

            // create the input matrix
            var inputMatrix = _lap.CreateMatrix(numInputRows, _blockSize, false);
            for (uint i = 0; i < numInputRows; i++) {
                var sequenceIndex = i / batchSize;
                var batchIndex = i % batchSize;
                var inputRow = inputMatrix.GetRow(i);
                inputs[sequenceIndex].GetRow(batchIndex).CopyTo(inputRow, sourceOffset:0, targetOffset:0);
                encoderStates?[sequenceIndex].GetRow(batchIndex).CopyTo(inputRow, 0, _inputSize);
                decoderHiddenState?.GetRow(batchIndex).CopyTo(inputRow, 0, _inputSize + _encoderSize);
            }

            // find the per batch softmax
            using var output = inputMatrix.TransposeAndMultiply(_attention);
            using var output2 = output.Reshape(batchSize, previousBatch.SequenceCount);
            var softmax = output2.SoftmaxPerRow();

            // construct the final attention output by multiplying attention weights
            using var weightMatrix = _lap.CreateMatrix(inputMatrix.RowCount, inputMatrix.ColumnCount, (i, _) => softmax[i / sequenceSize][i % sequenceSize]);
            using var outputMatrix = inputMatrix.PointwiseMultiply(weightMatrix);
            using var outputTensor = outputMatrix.Reshape(null, batchSize, _blockSize);
            using var combined = outputTensor.AddAllMatrices();
            combined.MultiplyInPlace(1f / outputTensor.Depth);
            combined.ConstrainInPlace(-10f, 10f);

            // append the attention output to the existing input signal
            var final = signal.GetMatrix().ConcatRight(combined);
            return (this, final.AsGraphData(), () => new Backpropagation(this, signal.Columns, sequenceSize, inputMatrix, softmax));
        }

        public override void ApplyError(NodeErrorType type, ITensor<float> delta, ILearningContext context)
        {
            _updater.Update(_attention, (IMatrix<float>)delta, context);
        }

        protected override (string Description, byte[] Data) GetInfo()
        {
            return ("SA", WriteData(WriteTo));
        }

        public override void WriteTo(BinaryWriter writer)
        {
            _encoderName.WriteTo(writer);
            _decoderName.WriteTo(writer);
            _inputSize.WriteTo(writer);
            _encoderSize.WriteTo(writer);
            _decoderSize.WriteTo(writer);
            _attention.WriteTo(writer);
        }

        public override void ReadFrom(GraphFactory factory, BinaryReader reader)
        {
            _lap = factory.LinearAlgebraProvider;
            _encoderName = reader.ReadString();
            _decoderName = reader.ReadString();
            _inputSize = reader.ReadUInt32();
            _encoderSize = reader.ReadUInt32();
            _decoderSize = reader.ReadUInt32();
            _blockSize = _inputSize + _encoderSize + _decoderSize;

            var attention = factory.Context.ReadMatrixFrom(reader);
            if (_attention == null)
                _attention = attention.Create(_lap);
            else
                attention.CopyTo(_attention);
            _updater ??= factory.CreateWeightUpdater(_attention);
        }
    }
}
