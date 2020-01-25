﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using BrightData;
using BrightTable.Buffers;
using BrightTable.Builders;
using BrightTable.Input;
using BrightTable.Segments;
using BrightTable.Transformations;

namespace BrightTable
{
    class RowOrientedDataTable : DataTableBase, IRowOrientedDataTable
    {
        readonly ColumnInfo[] _columns;
        readonly ColumnType[] _columnTypes;
        readonly uint[] _rowOffset;
        readonly InputData _data;
        readonly Func<BinaryReader, object>[] _columnReaders;

        public RowOrientedDataTable(IBrightDataContext context, InputData data, bool readHeader) : base(context)
        {
            _data = data;
            var reader = data.Reader;

            if (readHeader) {
                var version = reader.ReadInt32();
                if (version > Consts.DataTableVersion)
                    throw new Exception($"Data table version {version} exceeds {Consts.DataTableVersion}");
                var orientation = (DataTableOrientation) reader.ReadInt32();
                if (orientation != DataTableOrientation.RowOriented)
                    throw new Exception("Invalid orientation");
            }

            var numColumns = reader.ReadUInt32();
            _columns = new ColumnInfo[numColumns];
            for (uint i = 0; i < numColumns; i++)
                _columns[i] = new ColumnInfo(reader, i);
            _columnTypes = _columns.Select(c => c.ColumnType).ToArray();

            uint rowCount = reader.ReadUInt32();
            _rowOffset = new uint[rowCount];
            for (uint i = 0; i < rowCount; i++)
                _rowOffset[i] = reader.ReadUInt32();

            RowCount = (uint)_rowOffset.Length;
            ColumnCount = (uint)_columns.Length;

            _columnReaders = _columnTypes.Select(ct => _GetReader(ct)).ToArray();
        }

        public void Dispose()
        {
            _data.Dispose();
        }

        public DataTableOrientation Orientation => DataTableOrientation.RowOriented;
        public IReadOnlyList<ColumnType> ColumnTypes => _columnTypes;

        public IReadOnlyList<IDataTableSegment> Rows(params uint[] rowIndices)
        {
            var ret = new List<IDataTableSegment>();
            if (rowIndices.Any()) {
                ForEachRow(rowIndices, row => ret.Add(new Row(_columnTypes, row)));
            }

            return ret;
        }

        public IDataTableSegment Row(uint rowIndex) => Rows(rowIndex).Single();

        public void ForEachRow(IEnumerable<uint> rowIndices, Action<object[]> callback)
        {
            var row = new object[_columns.Length];
            lock (_data) {
                var reader = _data.Reader;
                foreach (var index in rowIndices) {
                    _data.MoveTo(_rowOffset[index]);
                    for (int i = 0, len = _columnReaders.Length; i < len; i++)
                        row[i] = _columnReaders[i](reader);
                    callback(row);
                }
            }
        }

        //public ISingleTypeTableSegment Column(uint columnIndex)
        //{
        //    return Columns(columnIndex).Single();
        //}

        public IReadOnlyList<ISingleTypeTableSegment> Columns(params uint[] columnIndices)
        {
            // TODO: optionally compress the columns based on unique count statistics
            var columns = columnIndices.Select(i => (Index: i, Column: _GetColumn(_columnTypes[i], _columns[i].MetaData))).ToList();
            if (columns.Any()) {
                // set the column metadata
                columns.ForEach(item => {
                    var metadata = item.Column.Segment.MetaData;
                    var column = _columns[item.Index];
                    column.MetaData.CopyTo(metadata);
                });

                // copy the column data
                ForEachRow((row, index) => {
                    foreach (var column in columns)
                        column.Column.Buffer.Set(index, row[column.Index]);
                });
            }

            return columns.Select(c => c.Column.Segment).ToList();
        }

        public IReadOnlyList<IMetaData> ColumnMetaData(params uint[] columnIndices)
        {
            return columnIndices.Select(i => _columns[i].MetaData).ToList().AsReadOnly();
        }

        public IRowOrientedDataTable AsRowOriented(string filePath = null)
        {
            using var builder = new RowOrientedTableBuilder(RowCount, filePath);

            foreach (var column in _columns)
                builder.AddColumn(column.ColumnType, column.MetaData);

            // ReSharper disable once AccessToDisposedClosure
            ForEachRow((row, index) => builder.AddRow(row));

            return builder.Build(Context);
        }

        public void ForEachRow(Action<object[], uint> callback, uint maxRows = uint.MaxValue)
        {
            var row = new object[ColumnCount];
            var rowCount = Math.Min(maxRows, RowCount);

            lock (_data) {
                _data.MoveTo(_rowOffset[0]);
                var reader = _data.Reader;

                for (uint i = 0; i < rowCount; i++) {
                    for (int j = 0, len = _columnReaders.Length; j < len; j++)
                        row[j] = _columnReaders[j](reader);
                    callback(row, i);
                }
            }
        }

        public IColumnOrientedDataTable AsColumnOriented(string filePath = null)
        {
            var columnOffsets = new List<(long Position, long EndOfColumnOffset)>();
            using var builder = new ColumnOrientedTableBuilder(filePath);

            builder.WriteHeader(ColumnCount, RowCount);
            var columns = Columns(Enumerable.Range(0, _columns.Length).Select(i => (uint)i).ToArray());
            foreach(var column in columns) {
                var position = builder.Write(column);
                columnOffsets.Add((position, builder.GetCurrentPosition()));
            }
            builder.WriteColumnOffsets(columnOffsets);
            return builder.Build(Context);
        }

        (ISingleTypeTableSegment Segment, IEditableBuffer Buffer) _GetColumn(ColumnType columnType, IMetaData metadata)
        {
            var type = typeof(DataSegmentBuffer<>).MakeGenericType(columnType.GetColumnType());
            var ret = Activator.CreateInstance(type, Context, columnType, metadata, RowCount);
            return ((ISingleTypeTableSegment) ret, (IEditableBuffer) ret);
        }

        private object _ReadString(BinaryReader reader) => reader.ReadString();
        private object _ReadDouble(BinaryReader reader) => reader.ReadDouble();
        private object _ReadDecimal(BinaryReader reader) => reader.ReadDecimal();
        private object _ReadInt32(BinaryReader reader) => reader.ReadInt32();
        private object _ReadInt16(BinaryReader reader) => reader.ReadInt16();
        private object _ReadSingle(BinaryReader reader) => reader.ReadSingle();
        private object _ReadBoolean(BinaryReader reader) => reader.ReadBoolean();
        private object _ReadDate(BinaryReader reader) => new DateTime(reader.ReadInt64());
        private object _ReadInt64(BinaryReader reader) => reader.ReadInt64();
        private object _ReadByte(BinaryReader reader) => reader.ReadSByte();
        private object _ReadIndexList(BinaryReader reader) => IndexList.ReadFrom(Context, reader);
        private object _ReadWeightedIndexList(BinaryReader reader) => WeightedIndexList.ReadFrom(Context, reader);
        private object _ReadVector(BinaryReader reader) => new Vector<float>(Context, reader);
        private object _ReadMatrix(BinaryReader reader) => new Matrix<float>(Context, reader);
        private object _ReadTensor3D(BinaryReader reader) => new Tensor3D<float>(Context, reader);
        private object _ReadTensor4D(BinaryReader reader) => new Tensor4D<float>(Context, reader);
        private object _ReadBinaryData(BinaryReader reader) => new BinaryData(reader);

        Func<BinaryReader, object> _GetReader(ColumnType type)
		{
			switch (type) {
				case ColumnType.String:
					return _ReadString;
				case ColumnType.Double:
					return _ReadDouble;
				case ColumnType.Decimal:
					return _ReadDecimal;
				case ColumnType.Int:
					return _ReadInt32;
				case ColumnType.Short:
					return _ReadInt16;
				case ColumnType.Float:
					return _ReadSingle;
				case ColumnType.Boolean:
					return _ReadBoolean;
				case ColumnType.Date:
                    return _ReadDate;
                case ColumnType.Long:
					return _ReadInt64;
				case ColumnType.Byte:
					return _ReadByte;
				case ColumnType.IndexList:
                    return _ReadIndexList;
                case ColumnType.WeightedIndexList:
					return _ReadWeightedIndexList;
				case ColumnType.Vector:
                    return _ReadVector;
                case ColumnType.Matrix:
					return _ReadMatrix;
				case ColumnType.Tensor3D:
                    return _ReadTensor3D;
				case ColumnType.Tensor4D:
					return _ReadTensor4D;
                case ColumnType.BinaryData:
                    return _ReadBinaryData;
				default:
					return null;
			}
		}

        public IRowOrientedDataTable Bag(uint sampleCount, int? randomSeed = null, string filePath = null)
        {
            var conversion = new BagTableTransformation(sampleCount, randomSeed);
            return conversion.Transform(this, filePath);
        }

        public IRowOrientedDataTable Concat(params IRowOrientedDataTable[] others) => Concat(null, others);
        public IRowOrientedDataTable Concat(string filePath, params IRowOrientedDataTable[] others)
        {
            var t = new ConcatTablesTransformation(new [] {this}.Concat(others).ToArray());
            return t.Transform(this, filePath);
        }

        public IRowOrientedDataTable Mutate(Func<object[], object[]> projector, string filePath = null)
        {
            var t = new ProjectRowsTransformation(projector);
            return t.Transform(this, filePath);
        }

        public IRowOrientedDataTable SelectRows(params uint[] rowIndices) => SelectRows(null, rowIndices);
        public IRowOrientedDataTable SelectRows(string filePath, params uint[] rowIndices)
        {
            var t = new RowSubsetTransformation(rowIndices);
            return t.Transform(this, filePath);
        }

        public IRowOrientedDataTable Shuffle(int? randomSeed = null, string filePath = null)
        {
            var t = new ShuffleTableTransformation(randomSeed);
            return t.Transform(this, filePath);
        }

        public IRowOrientedDataTable Sort(bool ascending, uint columnIndex, string filePath = null)
        {
            var t = new SortTableTransformation(ascending, columnIndex);
            return t.Transform(this, filePath);
        }

        public IRowOrientedDataTable Vectorise(string columnName, params uint[] vectorColumnIndices) => Vectorise(null, columnName, vectorColumnIndices);
        public IRowOrientedDataTable Vectorise(string filePath, string columnName, params uint[] vectorColumnIndices)
        {
            var t = new VectoriseTableTransformation(columnName, vectorColumnIndices);
            return t.Transform(this, filePath);
        }
    }
}
