﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BrightData.Buffer.Operations;
using BrightData.DataTable.ConstraintValidation;
using BrightData.LinearAlgebra.ReadOnly;
using BrightData.Types;

namespace BrightData.DataTable
{
    /// <summary>
    /// Builds a data table dynamically
    /// </summary>
    internal class ColumnOrientedDataTableBuilder(
        BrightDataContext context,
        IProvideDataBlocks? tempData = null,
        int blockSize = Consts.DefaultBlockSize,
        uint? maxInMemoryBlocks = Consts.DefaultMaxBlocksInMemory)
        : IBuildDataTables
    {
        readonly List<ICompositeBuffer> _columns = [];

        public MetaData TableMetaData { get; } = new();
        public MetaData[] ColumnMetaData => _columns.Select(x => x.MetaData).ToArray();
        public uint RowCount { get; private set; }
        public uint ColumnCount => (uint)_columns.Count;
        public BrightDataContext Context { get; } = context;

        internal static string DefaultColumnName(string? name, int numColumns)
        {
            return String.IsNullOrWhiteSpace(name) ? $"Column {numColumns + 1}" : name;
        }

        public ICompositeBuffer CreateColumn(BrightDataType type, string? name = null)
        {
            var columnMetaData = new MetaData();
            columnMetaData.Set(Consts.Name, DefaultColumnName(name, _columns.Count));
            columnMetaData.Set(Consts.Type, type);
            columnMetaData.Set(Consts.ColumnIndex, (uint)_columns.Count);
            if (type.IsNumeric())
                columnMetaData.Set(Consts.IsNumeric, true);

            return CreateColumn(type, columnMetaData);
        }

        public ICompositeBuffer CreateColumn(BrightDataType type, MetaData metaData)
        {
            var buffer = type.CreateCompositeBuffer(tempData, blockSize, maxInMemoryBlocks);
            metaData.CopyTo(buffer.MetaData);
            buffer.MetaData.Set(Consts.ColumnIndex, (uint)_columns.Count);
            _columns.Add(buffer);
            return buffer;
        }

        public ICompositeBuffer[] CreateColumnsFrom(IDataTable table, params uint[] columnIndices)
        {
            var columnSet = new HashSet<uint>(table.AllOrSpecifiedColumnIndices(false, columnIndices));
            var columnTypes = table.ColumnTypes.Zip(table.ColumnMetaData, (t, m) => (Type: t, MetaData: m))
                .Select((c, i) => (Column: c, Index: (uint) i));

            var wantedColumnTypes = columnTypes
                .Where(c => columnSet.Contains(c.Index))
                .Select(c => c.Column)
                .ToList()
            ;

            var index = 0;
            var ret = new ICompositeBuffer[wantedColumnTypes.Count];
            foreach (var column in wantedColumnTypes)
                ret[index++] = CreateColumn(column.Type, column.MetaData);
            return ret;
        }

        public ICompositeBuffer CreateColumn(IReadOnlyBufferWithMetaData buffer)
        {
            return CreateColumn(buffer.DataType.GetBrightDataType(), buffer.MetaData);
        }

        public ICompositeBuffer[] CreateColumnsFrom(params IReadOnlyBufferWithMetaData[] buffers)
        {
            return buffers.Select(x => CreateColumn(x.DataType.GetBrightDataType(), x.MetaData)).ToArray();
        }

        public ICompositeBuffer[] CreateColumnsFrom(IEnumerable<IReadOnlyBufferWithMetaData> buffers)
        {
            return buffers.Select(x => CreateColumn(x.DataType.GetBrightDataType(), x.MetaData)).ToArray();
        }

        public ICompositeBuffer<T> CreateColumn<T>(string? name = null)
            where T : notnull
        {
            var type = typeof(T).GetBrightDataType();
            return (ICompositeBuffer<T>)CreateColumn(type, name);
        }

        public void AddRow(params object[] items)
        {
            for(int i = 0, len = items.Length; i < len; i++)
                _columns[i].AppendObject(items[i]);
            ++RowCount;
        }

        public Task AddRows(IReadOnlyList<IReadOnlyBuffer> buffers, CancellationToken ct = default)
        {
            var copy = new ManyToManyCopy(buffers, _columns);
            return copy.Execute(null, null, ct).ContinueWith(_ => RowCount += copy.CopiedCount, ct);
        }

        public Task AddRows(IReadOnlyList<IReadOnlyBufferWithMetaData> buffers, CancellationToken ct = default)
        {
            var copy = new ManyToManyCopy(buffers, _columns);
            return copy.Execute(null, null, ct).ContinueWith(_ => RowCount += copy.CopiedCount, ct);
        }

        public Task WriteTo(Stream stream)
        {
            var writer = new ColumnOrientedDataTableWriter(tempData, blockSize, maxInMemoryBlocks);
            return writer.Write(
                TableMetaData,
                _columns.Cast<IReadOnlyBufferWithMetaData>().ToArray(),
                stream
            );
        }

        public ICompositeBuffer<ReadOnlyVector> CreateFixedSizeVectorColumn(uint size, string? name)
        {
            var ret = CreateColumn<ReadOnlyVector>(name);
            ret.ConstraintValidator = new ThrowOnInvalidConstraint<ReadOnlyVector>((in ReadOnlyVector tensor) => tensor.Size == size);
            return ret;
        }

        
        public ICompositeBuffer<ReadOnlyMatrix> CreateFixedSizeMatrixColumn(uint rows, uint columns, string? name)
        {
            var ret = CreateColumn<ReadOnlyMatrix>(name);
            ret.ConstraintValidator = new ThrowOnInvalidConstraint<ReadOnlyMatrix>((in ReadOnlyMatrix tensor) => 
                tensor.RowCount == rows 
                && tensor.ColumnCount == columns 
            );
            return ret;
        }

        public ICompositeBuffer<ReadOnlyTensor3D> CreateFixedSize3DTensorColumn(uint depth, uint rows, uint columns, string? name)
        {
            var ret = CreateColumn<ReadOnlyTensor3D>(name);
            ret.ConstraintValidator = new ThrowOnInvalidConstraint<ReadOnlyTensor3D>((in ReadOnlyTensor3D tensor) => 
                tensor.RowCount == rows 
                && tensor.ColumnCount == columns 
                && tensor.Depth == depth
            );
            return ret;
        }

        public ICompositeBuffer<ReadOnlyTensor4D> CreateFixedSize4DTensorColumn(uint count, uint depth, uint rows, uint columns, string? name)
        {
            var ret = CreateColumn<ReadOnlyTensor4D>(name);
            ret.ConstraintValidator = new ThrowOnInvalidConstraint<ReadOnlyTensor4D>((in ReadOnlyTensor4D tensor) => 
                tensor.RowCount == rows 
                && tensor.ColumnCount == columns 
                && tensor.Depth == depth 
                && tensor.Count == count
            );
            return ret;
        }
    }
}
