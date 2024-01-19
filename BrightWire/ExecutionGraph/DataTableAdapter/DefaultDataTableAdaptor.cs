﻿using System.Linq;
using BrightData;
using BrightData.DataTable;

namespace BrightWire.ExecutionGraph.DataTableAdapter
{
    /// <summary>
    /// Vectorise each row of the data table on demand
    /// </summary>
    internal class DefaultDataTableAdapter : RowBasedDataTableAdapterBase
    {
        readonly uint[] _featureColumns;

        public DefaultDataTableAdapter(IDataTable dataTable, VectorisationModel? inputVectoriser, VectorisationModel? outputVectoriser, uint[] featureColumns)
            : base(dataTable, featureColumns)
        {
            _featureColumns = featureColumns;
            InputVectoriser = inputVectoriser ?? dataTable.GetVectoriser(true, _featureColumnIndices).Result;
            OutputVectoriser = outputVectoriser ?? dataTable.GetVectoriser(true, dataTable.GetTargetColumnOrThrow()).Result;
        }

        public override IDataSource CloneWith(IDataTable dataTable)
        {
            return new DefaultDataTableAdapter(dataTable, InputVectoriser, OutputVectoriser, _featureColumns);
        }

        public override uint InputSize => InputVectoriser!.OutputSize;
        public override uint? OutputSize => OutputVectoriser?.OutputSize;

        public override IMiniBatch Get(uint[] rowIndices)
        {
            var rows = GetRows(rowIndices);
            var data = rows
                .Select(r => (InputVectoriser!.Vectorise(r), OutputVectoriser!.Vectorise(r)))
                .ToArray()
            ;
            return GetMiniBatch(rowIndices, data);
        }
    }
}
