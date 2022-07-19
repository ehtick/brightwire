﻿using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using BrightWire.ExecutionGraph.Helper;
using BrightData;
using BrightData.LinearAlgebra;
using BrightDataTable = BrightData.DataTable.BrightDataTable;

namespace BrightWire.ExecutionGraph.DataTableAdapter
{
    /// <summary>
    /// Base class for data table based data adapters
    /// </summary>
    /// <typeparam name="T">The type of the cached data</typeparam>
    public abstract class DataTableAdapterBase<T> : IDataSource
    {
        /// <summary>
        /// The data table column indices with features
        /// </summary>
        protected readonly uint[] _featureColumnIndices;

		/// <summary>
		/// Target column index
		/// </summary>
        protected readonly uint _targetColumnIndex;

		/// <summary>
		/// Data table
		/// </summary>
        protected readonly BrightDataTable _dataTable;

        /// <summary>
	    /// Constructor
	    /// </summary>
        /// <param name="dataTable"></param>
	    /// <param name="featureColumns"></param>
	    protected DataTableAdapterBase(BrightDataTable dataTable, uint[] featureColumns)
        {
            _dataTable = dataTable;
            _targetColumnIndex = dataTable.GetTargetColumnOrThrow();
            _featureColumnIndices = featureColumns;
            RowCount = dataTable.RowCount;
        }

	    /// <inheritdoc />
        public abstract uint InputSize { get; }
	    /// <inheritdoc />
        public abstract uint? OutputSize { get; }
	    /// <inheritdoc />
        public virtual uint RowCount { get; }

        /// <inheritdoc />
        public abstract IMiniBatch Get(uint[] rows);

        /// <inheritdoc />
        public abstract IDataSource CloneWith(BrightDataTable dataTable);

        /// <inheritdoc />
        public IDataTableVectoriser? InputVectoriser { get; protected set; } = null;

        /// <inheritdoc />
        public IDataTableVectoriser? OutputVectoriser { get; protected set; } = null;

        /// <inheritdoc />
        public virtual uint[][] GetSequentialBatches()
        {
            return new[] {
                RowCount.AsRange().ToArray()
            };
        }

        /// <summary>
        /// Returns data for the specified row indices
        /// </summary>
        /// <param name="rows">List of row indices</param>
        protected abstract IEnumerable<T> GetRows(uint[] rows);

        /// <summary>
		/// Creates a mini batch
		/// </summary>
		/// <param name="rows">Row indices</param>
		/// <param name="data">Array of input/output pairs</param>
        protected IMiniBatch GetMiniBatch(uint[] rows, (float[] Input, float[] Output)[] data)
        {
            var lap = _dataTable.Context.LinearAlgebraProvider;
            var input = lap.CreateMatrix((uint)data.Length, InputSize, (x, y) => data[x].Input[y]).AsGraphData();
            var output = OutputSize > 0 
                ? lap.CreateMatrix((uint)data.Length, (uint)OutputSize, (x, y) => data[x].Output[y]).AsGraphData()
                : null;

            return new MiniBatch(rows, this, input, output);
        }

		/// <summary>
		/// Creates a sequential mini batch
		/// </summary>
		/// <param name="rows">Row indices</param>
		/// <param name="data">List of input/output matrix tuples</param>
        protected IMiniBatch GetSequentialMiniBatch(uint[] rows, (IReadOnlyMatrix Input, IReadOnlyMatrix? Output)[] data)
        {
            List<IReadOnlyVector>? temp;
            var inputData = new Dictionary<uint, List<IReadOnlyVector>>();
            var outputData = new Dictionary<uint, List<IReadOnlyVector>>();
            var lap = _dataTable.Context.LinearAlgebraProvider;

            foreach (var (input, output) in data) {
                for (uint i = 0, len = input.RowCount; i < len; i++) {
                    if (!inputData.TryGetValue(i, out temp))
                        inputData.Add(i, temp = new List<IReadOnlyVector>());
                    temp.Add(input.GetRow(i));

                    if (output != null) {
                        if (!outputData.TryGetValue(i, out temp))
                            outputData.Add(i, temp = new List<IReadOnlyVector>());
                        temp.Add(output.GetRow(i));
                    }
                }
            }

            var miniBatch = new MiniBatch(rows, this);
            foreach (var item in inputData.OrderBy(kv => kv.Key)) {
                var span = CollectionsMarshal.AsSpan(item.Value);
                var input = lap.CreateMatrixFromRows(span);
                IGraphData? output = null;
                if (outputData.TryGetValue(item.Key, out var outputList))
                    output = lap.CreateMatrixFromRows(CollectionsMarshal.AsSpan(outputList)).AsGraphData();
                var type = (item.Key == 0)
                    ? MiniBatchSequenceType.SequenceStart
                    : item.Key == (inputData.Count - 1)
                        ? MiniBatchSequenceType.SequenceEnd
                        : MiniBatchSequenceType.Standard
                ;
                miniBatch.Add(type, input.AsGraphData(), output);
            }
            return miniBatch;
        }
    }
}
