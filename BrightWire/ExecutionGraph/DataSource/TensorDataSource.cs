﻿using BrightData;
using BrightTable;
using BrightWire.ExecutionGraph.Helper;
using BrightWire.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BrightWire.ExecutionGraph.DataSource
{
    class TensorDataSource : IDataSource
    {
        readonly uint _rows, _columns, _depth, _matrixSize;
        readonly IReadOnlyList<Tensor3D<float>> _data;
        readonly ILinearAlgebraProvider _lap;

        public TensorDataSource(ILinearAlgebraProvider lap, IReadOnlyList<Tensor3D<float>> data)
        {
            _lap = lap;
            _data = data;

            var first = data.First();
            InputSize = first.Size;
            OutputSize = null;
            _rows = first.RowCount;
            _columns = first.ColumnCount;
            _depth = first.Depth;
            _matrixSize = _rows * _columns;
        }

        public bool IsSequential => false;
        public uint InputSize { get; }
	    public uint? OutputSize { get; }
	    public uint InputCount => 1;
        public uint RowCount => (uint)_data.Count;

        public IDataSource CloneWith(IRowOrientedDataTable dataTable)
        {
            throw new NotImplementedException();
        }

        public IMiniBatch Get(IExecutionContext executionContext, IReadOnlyList<uint> rows)
        {
            var data = rows.Select(i => _data[(int)i]).ToList();
            var input = _lap.CreateMatrix((uint)InputSize, (uint)data.Count, (i, j) => {
                var tensor = _data[(int)j];
                var rem = i % _matrixSize;
                var z = i / _matrixSize;
                var x = rem % _rows;
                var y = rem / _rows;
                return tensor.Matrix(z).Row(x).Segment[y];
            });

            var inputList = new List<IGraphData> {
                new Tensor4DGraphData(input, _rows, _columns, _depth)
            };
            return new MiniBatch(rows, this, inputList, null);
        }

        public IReadOnlyList<IReadOnlyList<uint>> GetBuckets()
        {
            return new[] {
                _data.Count.AsRange().ToList()
            };
        }

        public void OnBatchProcessed(IContext context)
        {
            // nop
        }
    }
}
