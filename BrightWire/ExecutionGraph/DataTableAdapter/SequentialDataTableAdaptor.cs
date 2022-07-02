﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using BrightData;
using BrightData.DataTable2;
using BrightData.LinearAlgebra;
using BrightWire.ExecutionGraph.Helper;

namespace BrightWire.ExecutionGraph.DataTableAdapter
{
    /// <summary>
    /// Adapts data tables that classify each step of a sequence
    /// </summary>
    internal class SequentialDataTableAdapter : DataTableAdapterBase<(IMatrixInfo Input, IMatrixInfo? Output)>
    {
        readonly uint[] _featureColumns;
        readonly uint[] _rowDepth;
        readonly bool _sequenceLengthsAreVaried = false;

	    public SequentialDataTableAdapter(BrightDataTable dataTable, uint[] featureColumns, bool sequenceLengthsAreVaried = false) 
            : base(dataTable, featureColumns)
        {
            if (_featureColumnIndices.Length > 1)
                throw new NotImplementedException("Sequential datasets not supported with more than one input data column");
            if (dataTable.TableMetaData.Get("Seq2Seq", false))
                sequenceLengthsAreVaried = true;
            _featureColumns = featureColumns;

            _rowDepth = new uint[dataTable.RowCount];

            // find the number of sequences of each row
            var foundData = false;
            foreach(var (i, row) in dataTable.GetAllRowData()) {
                var inputMatrix = (IMatrixInfo) row[_featureColumnIndices[0]];
                var outputMatrix = (IMatrixInfo) row[_targetColumnIndex];
                _rowDepth[i] = inputMatrix.RowCount;
                if (outputMatrix.RowCount != inputMatrix.RowCount)
                    sequenceLengthsAreVaried = true;
                if (!foundData) {
                    InputSize = inputMatrix.ColumnCount;
                    OutputSize = outputMatrix.ColumnCount;
                    foundData = true;
                }
            }
            if (!foundData)
                throw new Exception("No data found");

            _sequenceLengthsAreVaried = sequenceLengthsAreVaried;
        }

        protected override IEnumerable<(IMatrixInfo Input, IMatrixInfo? Output)> GetRows(uint[] rows)
        {
            return _dataTable.GetRows(rows).Select(row => {
                using(row) return ((IMatrixInfo)row[_featureColumnIndices[0]], (IMatrixInfo?)row[_targetColumnIndex]);
            });
        }

        public override IDataSource CloneWith(BrightDataTable dataTable)
        {
            return new SequentialDataTableAdapter(dataTable, _featureColumns, _sequenceLengthsAreVaried);
        }

        public override uint InputSize { get; }
	    public override uint? OutputSize { get; }
	    public override uint RowCount => (uint)_rowDepth.Length;

        public override IMiniBatch Get(uint[] rows)
        {
            var lap = _dataTable.Context.LinearAlgebraProvider2;
            if (_sequenceLengthsAreVaried) {
                var inputData = new Dictionary<uint, List<IVectorInfo>>();
                var outputData = new Dictionary<uint, List<IVectorInfo>>();

                foreach (var (input, output) in GetRows(rows)) {
                    for (uint i = 0, len = input.RowCount; i < len; i++) {
                        if (!inputData.TryGetValue(i, out var temp))
                            inputData.Add(i, temp = new());
                        temp.Add(input.GetRow(i));
                    }

                    if (output != null) {
                        for (uint i = 0, len = output.RowCount; i < len; i++) {
                            if (!outputData.TryGetValue(i, out var temp))
                                outputData.Add(i, temp = new());
                            temp.Add(output.GetRow(i));
                        }
                    }
                }

                var encoderMiniBatch = new MiniBatch(rows, this);
                foreach (var item in inputData.OrderBy(kv => kv.Key)) {
                    var input = lap.CreateMatrixFromRows(CollectionsMarshal.AsSpan(item.Value));
                    var type = item.Key == 0
                        ? MiniBatchSequenceType.SequenceStart
                        : item.Key == (inputData.Count - 1)
                            ? MiniBatchSequenceType.SequenceEnd
                            : MiniBatchSequenceType.Standard
                    ;
                    encoderMiniBatch.Add(type, input.AsGraphData(), null);
                }

                var decoderMiniBatch = new MiniBatch(rows, this);
                foreach (var item in outputData.OrderBy(kv => kv.Key)) {
                    var output = lap.CreateMatrixFromRows(CollectionsMarshal.AsSpan(item.Value));
                    var type = item.Key == 0
                        ? MiniBatchSequenceType.SequenceStart
                        : item.Key == (outputData.Count - 1)
                            ? MiniBatchSequenceType.SequenceEnd
                            : MiniBatchSequenceType.Standard
                    ;
                    decoderMiniBatch.Add(type, null, output.AsGraphData());
                }

                encoderMiniBatch.NextMiniBatch = decoderMiniBatch;
                decoderMiniBatch.PreviousMiniBatch = encoderMiniBatch;
                return encoderMiniBatch;
            }
            return GetSequentialMiniBatch(rows, GetRows(rows).ToArray());
        }

        public override uint[][] GetSequentialBatches()
        {
            return _rowDepth
                .Select((r, i) => (Row: r, Index: i))
                .GroupBy(t => t.Row)
                .Select(g => g.Select(d => (uint)d.Index).ToArray())
                .ToArray()
            ;
        }
    }
}
