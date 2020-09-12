﻿using System.Collections.Generic;
using System.Linq;
using BrightData;
using BrightTable;
using BrightTable.Transformations;
using BrightWire.Models;

namespace BrightWire.ExecutionGraph.DataTableAdaptor
{
    /// <summary>
    /// Adapts data tables with a index list based column (corresponding to an unweighted sparse vector)
    /// </summary>
    class IndexListDataTableAdaptor : DataTableAdaptorBase<(List<IndexList>, Vector<float>)>, IIndexListEncoder
    {
        readonly uint _inputSize;
        readonly DataTableVectoriser _vectoriser;

        public IndexListDataTableAdaptor(ILinearAlgebraProvider lap, IRowOrientedDataTable dataTable, DataTableVectoriser vectoriser)
            : base(lap, dataTable)
        {
            _vectoriser = vectoriser ?? new DataTableVectoriser(dataTable);
            _inputSize = _vectoriser.InputSize;
            OutputSize = _vectoriser.OutputSize;

            // load the data
            //dataTable.ForEachRow(row => _segment.Add((_dataColumnIndex.Select(i => (IndexList)row[i]).ToList(), _vectoriser.GetOutput(row))));
        }

        public override bool IsSequential => false;
        public override uint InputSize => _inputSize;
        public override uint? OutputSize { get; }
	    public override uint RowCount => (uint)_data.Count;

        public float[] Encode(IndexList indexList)
        {
            var ret = new float[_inputSize];
            foreach (var group in indexList.Indices.GroupBy(d => d))
                ret[group.Key] = group.Count();
            return ret;
        }

        public override IMiniBatch Get(IExecutionContext executionContext, IReadOnlyList<uint> rows)
        {
            var data = _GetRows(rows)
                .Select(r => (r.Item1.Select(Encode).ToArray(), r.Item2.Segment.ToArray()))
                .ToList()
            ;
            return _GetMiniBatch(rows, data);
        }

        public override IDataSource CloneWith(IRowOrientedDataTable dataTable)
        {
            return new IndexListDataTableAdaptor(_lap, dataTable, _vectoriser);
        }
    }
}
