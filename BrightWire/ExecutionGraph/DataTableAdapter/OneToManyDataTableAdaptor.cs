﻿using BrightData;
using BrightWire.ExecutionGraph.Helper;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using BrightData.LinearAlgebra;
using BrightDataTable = BrightData.DataTable.BrightDataTable;

namespace BrightWire.ExecutionGraph.DataTableAdapter
{
    /// <summary>
    /// Adapts data tables that generate sequences from a single vector
    /// </summary>
    internal class OneToManyDataTableAdapter : RowBasedDataTableAdapterBase
    {
        readonly uint[] _featureColumns;
        readonly uint[] _rowDepth;

	    public OneToManyDataTableAdapter(BrightDataTable dataTable, uint[] featureColumns) 
            : base(dataTable, featureColumns)
        {
            if (_featureColumnIndices.Length > 1)
                throw new NotImplementedException("Sequential datasets not supported with more than one input data column");
            _featureColumns = featureColumns;

            // find the number of sequences of each row
            _rowDepth = new uint[dataTable.RowCount];
            IVectorInfo? inputVector = null;
            IMatrixInfo? outputMatrix = null;
            foreach(var (i, row) in dataTable.GetAllRowData()) {
                inputVector = (IVectorInfo)row[_featureColumnIndices[0]];
                outputMatrix = (IMatrixInfo)row[_targetColumnIndex];
                _rowDepth[i] = outputMatrix.RowCount;
                if (outputMatrix.ColumnCount != inputVector.Size)
                    throw new ArgumentException("Rows between input and output data tables do not match");
            }
            if (inputVector == null || outputMatrix == null)
                throw new Exception("No data found");

            InputSize = inputVector.Size;
            OutputSize = outputMatrix.ColumnCount;
        }

        public override IDataSource CloneWith(BrightDataTable dataTable)
        {
            return new OneToManyDataTableAdapter(dataTable, _featureColumns);
        }

        public override uint InputSize { get; }
	    public override uint? OutputSize { get; }

	    public override uint[][] GetSequentialBatches()
        {
            return _rowDepth
                .Select((r, i) => (Row: r, Index: i))
                .GroupBy(t => t.Row)
                .Select(g => g.Select(d => (uint)d.Index).ToArray())
                .ToArray()
            ;
        }

        public override IMiniBatch Get(uint[] rows)
        {
            var lap = _dataTable.Context.LinearAlgebraProvider;
            var data = GetRows(rows)
                .Select(r => ((IVectorInfo)r[_featureColumnIndices[0]], (IMatrixInfo)r[_targetColumnIndex]))
                .ToList()
            ;
            var outputData = new Dictionary<uint, List<IVectorInfo>>();
            foreach (var item in data) {
                var output = item.Item2;
                for (uint i = 0, len = output.RowCount; i < len; i++) {
                    if (!outputData.TryGetValue(i, out var temp))
                        outputData.Add(i, temp = new());
                    temp.Add(output.GetRow(i));
                }
            }

            var miniBatch = new MiniBatch(rows, this);
            var curr = lap.CreateMatrix((uint)data.Count, InputSize, (x, y) => data[(int)x].Item1.Segment[y]);
            foreach (var item in outputData.OrderBy(kv => kv.Key)) {
                var output = lap.CreateMatrixFromRows(CollectionsMarshal.AsSpan(item.Value));
                var type = (item.Key == 0)
                    ? MiniBatchSequenceType.SequenceStart
                    : item.Key == (outputData.Count - 1)
                        ? MiniBatchSequenceType.SequenceEnd
                        : MiniBatchSequenceType.Standard
                ;
                miniBatch.Add(type, curr.AsGraphData(), output.AsGraphData());
                curr = output;
            }
            return miniBatch;
        }
    }
}
