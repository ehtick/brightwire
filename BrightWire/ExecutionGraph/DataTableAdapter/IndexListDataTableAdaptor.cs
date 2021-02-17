﻿using System.Collections.Generic;
using System.Linq;
using BrightData;

namespace BrightWire.ExecutionGraph.DataTableAdapter
{
    /// <summary>
    /// Adapts data tables with a index list based column (corresponding to an unweighted sparse vector)
    /// </summary>
    internal class IndexListDataTableAdapter : DataTableAdapterBase<(IndexList, float[])>, IIndexListEncoder
    {
        readonly uint[] _featureColumns;

        public IndexListDataTableAdapter(ILinearAlgebraProvider lap, IRowOrientedDataTable dataTable, IDataTableVectoriser? outputVectoriser, uint[] featureColumns)
            : base(lap, dataTable, featureColumns)
        {
            _featureColumns = featureColumns;
            OutputVectoriser = outputVectoriser ?? dataTable.GetVectoriser(true, dataTable.GetTargetColumnOrThrow());
            OutputSize = OutputVectoriser.OutputSize;

            // load the data
            uint inputSize = 0;
            dataTable.ForEachRow(row => _data.Add((Combine(_featureColumnIndices.Select(i => (IndexList)row[i]), ref inputSize), OutputVectoriser.Vectorise(row))));
            InputSize = inputSize;
        }

        public override bool IsSequential => false;
        public override uint InputSize { get; }
        public override uint? OutputSize { get; }
        public override uint RowCount => (uint)_data.Count;

        public IndexList Combine(IEnumerable<IndexList> lists, ref uint inputSize)
        {
            var ret = IndexList.Merge(lists);
            var maxIndex = ret.Indices.Max();
            if (maxIndex > inputSize)
                inputSize = maxIndex + 1;
            return ret;
        }

        public float[] Encode(IndexList indexList)
        {
            var ret = new float[InputSize];
            foreach (var group in indexList.Indices.GroupBy(d => d))
                ret[group.Key] = group.Count();
            return ret;
        }

        public override IMiniBatch Get(uint[] rows)
        {
            var data = GetRows(rows).Select(r => (new[] { Encode(r.Item1)}, r.Item2)).ToArray();
            return GetMiniBatch(rows, data);
        }

        public override IDataSource CloneWith(IRowOrientedDataTable dataTable)
        {
            return new IndexListDataTableAdapter(_lap, dataTable, OutputVectoriser, _featureColumns);
        }
    }
}