﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using BrightData.Buffer.Composite;
using BrightData.Buffer.ReadOnly;
using BrightData.Buffer.ReadOnly.Helper;
using BrightData.Converter;
using BrightData.DataTable.Columns;
using BrightData.DataTable.Rows;
using BrightData.Helper;
using BrightData.LinearAlgebra.ReadOnly;
using BrightData.Types;
using CommunityToolkit.HighPerformance;

namespace BrightData.DataTable
{
    /// <summary>
    /// Column oriented data table
    /// </summary>
    internal partial class ColumnOrientedDataTable : IDataTable
    {
        const int ReaderBlockSize = 128;
        internal struct Column
        {
            public BrightDataType DataType;
            public uint DataTypeSize;
            public readonly override string ToString() => $"{DataType} ({DataTypeSize})";
        }
        protected readonly IByteBlockReader                         _reader;
        protected readonly TableHeader                              _header;
        protected readonly Column[]                                 _columns;
        protected readonly IReadOnlyBufferWithMetaData[]            _columnReader;
        protected readonly MetaData[]                               _columnMetaData;
        protected Lazy<Task<List<string>>>                          _strings;
        readonly Lazy<Task<ReadOnlyMemory<float>>>                  _tensors;
        readonly Lazy<Task<ReadOnlyMemory<byte>>>                   _data;
        readonly Lazy<Task<ReadOnlyMemory<uint>>>                   _indices;
        readonly Lazy<Task<ReadOnlyMemory<WeightedIndexList.Item>>> _weightedIndices;
        readonly Lazy<IReadOnlyBuffer<object>[]>                    _genericColumns;
        readonly MethodInfo                                         _createColumnReader;
        BlockMapper<DataRangeColumnType, ReadOnlyVector>            _vectorMapper;
        BlockMapper<MatrixColumnType, ReadOnlyMatrix>               _matrixMapper;
        BlockMapper<Tensor3DColumnType, ReadOnlyTensor3D>           _tensor3DMapper;
        BlockMapper<Tensor4DColumnType, ReadOnlyTensor4D>           _tensor4DMapper;

        ColumnOrientedDataTable(BrightDataContext context, TableHeader header, Column[] columns, ReadOnlyMemory<byte> metaData, IByteBlockReader reader)
        {
            Context = context;
            _reader = reader;
            _header = header;
            _columns = columns;
            if (_header.Orientation != DataTableOrientation.ColumnOriented)
                throw new ArgumentException("Expected to read a column oriented table");
            if (_header.ColumnCount == 0)
                throw new Exception("Expected data table to contain at least one column");
            if (_header.RowCount == 0)
                throw new Exception("Expected data table to contain at least one row");

            // default tensor creators
            _vectorMapper   = GetVectors;
            _matrixMapper   = GetMatrices;
            _tensor3DMapper = Get3DTensors;
            _tensor4DMapper = Get4DTensors;

            // get column types
            ColumnTypes = new BrightDataType[ColumnCount];
            for(var i = 0; i < ColumnCount; i++)
                ColumnTypes[i] = _columns[i].DataType;

            // read the meta data
            _columnMetaData = new MetaData[ColumnCount];
            var metadataReader = new BinaryReader(metaData.AsStream(), Encoding.UTF8, true);
            for (var i = 0; i < ColumnCount + 1; i++) {
                if (i == 0)
                    MetaData = new(metadataReader);
                else
                    _columnMetaData[i-1] = new(metadataReader);
            }
            MetaData ??= new();

            // create column readers
            var genericMethods  = GetType().GetGenericMethods();
            _createColumnReader = genericMethods[nameof(CreateColumnReader)];
            _columnReader       = new IReadOnlyBufferWithMetaData[ColumnCount];
            CreateColumnReaders();

            // create data readers
            _strings            = new(() => ReadStrings(_header.StringOffset, _header.StringSizeBytes));
            _tensors            = new(() => GetBlock<float>(_header.TensorOffset, _header.TensorSizeBytes));
            _data               = new(() => GetBlock<byte>(_header.DataOffset, _header.DataSizeBytes));
            _indices            = new(() => GetBlock<uint>(_header.IndexOffset, _header.IndexSizeBytes));
            _weightedIndices    = new(() => GetBlock<WeightedIndexList.Item>(_header.WeightedIndexOffset, _header.WeightedIndexSizeBytes));
            _genericColumns     = new(GetColumnsAsObjectBuffers);
        }

        public static async Task<IDataTable> Load(BrightDataContext context, IByteBlockReader reader)
        {
            var headerSize = Unsafe.SizeOf<TableHeader>();
            var header = (await reader.GetBlock(0, (uint)headerSize)).Span.Cast<byte, TableHeader>()[0];
            var columns = (await reader.GetBlock(header.InfoOffset, header.InfoSizeBytes)).Span.Cast<byte, Column>().ToArray();
            var metaData = await reader.GetBlock(header.MetaDataOffset, reader.Size - header.MetaDataOffset);
            return new ColumnOrientedDataTable(context, header, columns, metaData, reader);
        }

        public MetaData MetaData { get; }

        void CreateColumnReaders()
        {
            var prevOffset = _header.DataOffset;
            for (uint i = 1; i < _columns.Length; i++) {
                var prevColumnType = ColumnTypes[i - 1];
                var (prevType, prevSize) = prevColumnType.GetColumnType();
                var nextOffset = prevOffset + prevSize * RowCount;
                CreateColumnReader(_createColumnReader, prevColumnType, prevType, i - 1, prevOffset, nextOffset - prevOffset);
                prevOffset = nextOffset;
            }
            var lastColumnType = ColumnTypes[_columns.Length - 1];
            var (lastColumnDataType, _) = lastColumnType.GetColumnType();
            CreateColumnReader(_createColumnReader, lastColumnType, lastColumnDataType, (uint)_columns.Length - 1, prevOffset, _header.DataOffset + _header.DataSizeBytes - prevOffset);
        }

        async Task<List<string>> ReadStrings(uint headerStringOffset, uint headerStringSizeBytes)
        {
            var ret = new List<string>();
            if (headerStringSizeBytes > 0) {
                var data = await _reader.GetBlock(headerStringOffset, headerStringSizeBytes);
                StringCompositeBuffer.Decode(data.Span, str => ret.Add(new string(str)));
            }
            return ret;
        }

        async Task<ReadOnlyMemory<T>> GetBlock<T>(uint start, uint size) where T : unmanaged
        {
            var ret = await _reader.GetBlock(start, size);
            return ret.Cast<byte, T>();
        }

        protected ReadOnlyMemory<string> GetStrings(ReadOnlySpan<uint> indices)
        {
            var index = 0;
            var stringTable = _strings.Value.Result;
            var ret = new string[indices.Length];
            foreach (var item in indices)
                ret[index++] = stringTable[(int)item];
            return ret;
        }
        protected string GetString(uint index) => _strings.Value.Result[(int)index];

        protected ReadOnlyMemory<BinaryData> GetBinaryData(ReadOnlySpan<DataRangeColumnType> span)
        {
            var index = 0;
            var ret = new BinaryData[span.Length];
            foreach (ref readonly var item in span)
                ret[index++] = GetBinaryData(item);
            return ret;
        }
        protected BinaryData GetBinaryData(in DataRangeColumnType dataRange)
        {
            var data = _data.Value.Result.Span.Slice((int)dataRange.StartIndex, (int)dataRange.Size);
            return new BinaryData(data);
        }

        protected ReadOnlyMemory<ReadOnlyVector> GetVectors(ReadOnlySpan<DataRangeColumnType> span)
        {
            var index = 0;
            var ret = new ReadOnlyVector[span.Length];
            var data = GetTensorData();
            foreach (ref readonly var item in span)
                ret[index++] = new ReadOnlyVector(data.Slice((int)item.StartIndex, (int)item.Size));
            return ret;
        }

        protected ReadOnlyMemory<ReadOnlyMatrix> GetMatrices(ReadOnlySpan<MatrixColumnType> span)
        {
            var index = 0;
            var ret = new ReadOnlyMatrix[span.Length];
            var data = GetTensorData();
            foreach (ref readonly var item in span)
                ret[index++] = new ReadOnlyMatrix(data.Slice((int)item.StartIndex, (int)item.Size), item.RowCount, item.ColumnCount);
            return ret;
        }

        protected ReadOnlyMemory<ReadOnlyTensor3D> Get3DTensors(ReadOnlySpan<Tensor3DColumnType> span)
        {
            var index = 0;
            var ret = new ReadOnlyTensor3D[span.Length];
            var data = GetTensorData();
            foreach (ref readonly var item in span)
                ret[index++] = new ReadOnlyTensor3D(data.Slice((int)item.StartIndex, (int)item.Size), item.Depth, item.RowCount, item.ColumnCount);
            return ret;
        }

        protected ReadOnlyMemory<ReadOnlyTensor4D> Get4DTensors(ReadOnlySpan<Tensor4DColumnType> span)
        {
            var index = 0;
            var ret = new ReadOnlyTensor4D[span.Length];
            var data = GetTensorData();
            foreach (ref readonly var item in span)
                ret[index++] = new ReadOnlyTensor4D(data.Slice((int)item.StartIndex, (int)item.Size), item.Count, item.Depth, item.RowCount, item.ColumnCount);
            return ret;
        }

        protected ReadOnlyMemory<IndexList> GetIndexLists(ReadOnlySpan<DataRangeColumnType> span)
        {
            var index = 0;
            var ret = new IndexList[span.Length];
            var data = _indices.Value.Result;
            foreach (ref readonly var item in span)
                ret[index++] = new IndexList(data.Slice((int)item.StartIndex, (int)item.Size));
            return ret;
        }
        protected IndexList GetIndexList(in DataRangeColumnType dataRange)
        {
            var data = _indices.Value.Result.Slice((int)dataRange.StartIndex, (int)dataRange.Size);
            return new IndexList(data);
        }

        protected ReadOnlyMemory<WeightedIndexList> GetWeightedIndexLists(ReadOnlySpan<DataRangeColumnType> span)
        {
            var index = 0;
            var ret = new WeightedIndexList[span.Length];
            var data = _weightedIndices.Value.Result;
            foreach (ref readonly var item in span)
                ret[index++] = new WeightedIndexList(data.Slice((int)item.StartIndex, (int)item.Size));
            return ret;
        }
        protected WeightedIndexList GetWeightedIndexList(in DataRangeColumnType dataRange)
        {
            var data = _weightedIndices.Value.Result.Slice((int)dataRange.StartIndex, (int)dataRange.Size);
            return new WeightedIndexList(data);
        }

        public uint RowCount => _header.RowCount;
        public uint ColumnCount => _header.ColumnCount;
        public DataTableOrientation Orientation => _header.Orientation;
        public BrightDataType[] ColumnTypes { get; }
        public MetaData[] ColumnMetaData => _columnMetaData;

        ~ColumnOrientedDataTable()
        {
            InternalDispose();
        }

        public void Dispose()
        {
            GC.SuppressFinalize(this);
            InternalDispose();
        }

        void InternalDispose()
        {
            _reader.Dispose();
        }

        /// <summary>
        /// Saves current meta data into the underlying stream
        /// </summary>
        public void PersistMetaData()
        {
            using var tempBuffer = new MemoryStream();
            using var metaDataWriter = new BinaryWriter(tempBuffer, Encoding.UTF8, true);
            MetaData.WriteTo(metaDataWriter);
            foreach(var item in _columnMetaData)
                item.WriteTo(metaDataWriter);
            metaDataWriter.Flush();

            var memory = new Memory<byte>(tempBuffer.GetBuffer(), 0, (int)tempBuffer.Length);
            _reader.Update(_header.MetaDataOffset, memory);
        }

        public async Task<MetaData[]> GetColumnAnalysis(params uint[] columnIndices)
        {
            if (!this.AllOrSpecifiedColumnIndices(true, columnIndices).All(i => _columnMetaData[i].Get(Consts.HasBeenAnalysed, false))) {
                var operations = this.AllOrSpecifiedColumnIndices(true, columnIndices).Select(i => _columnMetaData[i].Analyse(false, GetColumn(i))).ToArray();
                await operations.ExecuteAllAsOne();
            }
            return this.AllOrSpecifiedColumnIndices(false, columnIndices).Select(x => _columnMetaData[x]).ToArray();
        }

        public async IAsyncEnumerable<GenericTableRow> EnumerateRows([EnumeratorCancellation] CancellationToken ct = default)
        {
            var size = _header.ColumnCount;
            var enumerators = new IAsyncEnumerator<object>[size];
            var currentTasks = new ValueTask<bool>[size];
            var isValid = true;

            for (uint i = 0; i < size; i++)
                enumerators[i] = GetColumn(i).EnumerateAll().GetAsyncEnumerator(ct);

            uint rowIndex = 0;
            while (!ct.IsCancellationRequested && isValid) {
                for (var i = 0; i < size; i++)
                    currentTasks[i] = enumerators[i].MoveNextAsync();
                for (var i = 0; i < size; i++) {
                    if (await currentTasks[i] != true) {
                        isValid = false;
                        break;
                    }
                }

                if (isValid) {
                    var curr = new object[size];
                    for (var i = 0; i < size; i++)
                        curr[i] = enumerators[i].Current;
                    yield return new GenericTableRow(this, rowIndex++, curr);
                }
            }
        }

        public IReadOnlyBufferWithMetaData[] GetColumns(params uint[] columnIndices)
        {
            IReadOnlyBufferWithMetaData[] ret;
            if (columnIndices.Length == 0) {
                ret = new IReadOnlyBufferWithMetaData[ColumnCount];
                for (uint i = 0; i < ColumnCount; i++)
                    ret[i] = GetColumn(i);
            }
            else {
                var ind = 0;
                ret = new IReadOnlyBufferWithMetaData[columnIndices.Length];
                foreach(var index in columnIndices)
                    ret[ind++] = GetColumn(index);
            }

            return ret;
        }

        public IReadOnlyBufferWithMetaData[] GetColumns(IEnumerable<uint> columnIndices)
        {
            return columnIndices.Select(GetColumn).ToArray();
        }

        public async Task WriteColumnsTo(Stream stream, params uint[] columnIndices)
        {
            var writer = new ColumnOrientedDataTableWriter();
            await writer.Write(MetaData, GetColumns(columnIndices), stream);
        }

        public ReadOnlyMemory<float> GetTensorData() => _tensors.Value.Result;

        public void SetTensorMappers(
            BlockMapper<DataRangeColumnType, ReadOnlyVector> vectorMapper,
            BlockMapper<MatrixColumnType, ReadOnlyMatrix> matrixMapper,
            BlockMapper<Tensor3DColumnType, ReadOnlyTensor3D> tensor3DMapper,
            BlockMapper<Tensor4DColumnType, ReadOnlyTensor4D> tensor4DMapper
        )
        {
            _vectorMapper = vectorMapper;
            _matrixMapper = matrixMapper;
            _tensor3DMapper = tensor3DMapper;
            _tensor4DMapper = tensor4DMapper;
            CreateColumnReaders();
        }

        public BrightDataContext Context { get; }

        public async Task WriteRowsTo(Stream stream, params uint[] rowIndices)
        {
            var writer = new ColumnOrientedDataTableBuilder(Context);
            var newColumns = writer.CreateColumnsFrom(this);
            var wantedRowIndices = rowIndices.Length > 0 ? rowIndices : RowCount.AsRange().ToArray();
            var operations = newColumns
                .Select((x, i) => GenericTypeMapping.IndexedCopyOperation(GetColumn((uint)i), x, wantedRowIndices))
                .ToArray();
            await operations.ExecuteAllAsOne();
            await writer.WriteTo(stream);
        }

        IReadOnlyBuffer<object>[] GetColumnsAsObjectBuffers()
        {
            var index = 0;
            var ret = new IReadOnlyBuffer<object>[ColumnCount];
            for (uint i = 0; i < ColumnCount; i++) {
                var column = GetColumn(i);
                ret[index++] = GenericTypeMapping.ToObjectConverter(column);
            }
            return ret;
        }

        public async Task<GenericTableRow[]> GetRows(params uint[] rowIndices)
        {
            if (rowIndices.Length == 0) {
                Array.Resize(ref rowIndices, (int)RowCount);
                for (uint i = 0; i < RowCount; i++)
                    rowIndices[i] = i;
            }

            var columns = _genericColumns.Value;
            var len = columns.Length;
            var blockSize = columns[0].BlockSize;
            Debug.Assert(columns.Skip(1).All(x => x.BlockSize == blockSize));

            var blocks = rowIndices.Select(x => (SourceIndex: x, BlockIndex: x / blockSize, RelativeBlockIndex: x % blockSize))
                .GroupBy(x => x.BlockIndex)
                .OrderBy(x => x.Key)
            ;
            var ret = rowIndices.Select(x => new GenericTableRow(this, x, new object[len])).ToArray();
            var retTable = ret.ToLookup(x => x.RowIndex);
            var tasks = new Task<ReadOnlyMemory<object>>[len];
            foreach (var block in blocks) {
                for(var i = 0; i < len; i++)
                    tasks[i] = columns[i].GetTypedBlock(block.Key);
                await Task.WhenAll(tasks);
                foreach (var (sourceIndex, _, relativeBlockIndex) in block) {
                    foreach (var row in retTable[sourceIndex]) {
                        var rowValues = row.Values;
                        for (var i = 0; i < len; i++)
                            rowValues[i] = tasks[i].Result.Span[(int)relativeBlockIndex];
                    }
                }
            }
            return ret;
        }

        public GenericTableRow this[uint index]
        {
            get
            {
                var columns = _genericColumns.Value;
                var fetchTasks = columns.Select(x => x.GetItem(index)).ToArray();
                Task.WhenAll(fetchTasks).Wait();
                return new GenericTableRow(this, index, fetchTasks.Select(x => x.Result).ToArray());
            }
        }

        public Task<T> Get<T>(uint columnIndex, uint rowIndex) where T: notnull
        {
            var column = _columnReader[columnIndex];
            if (column.DataType != typeof(T))
                throw new ArgumentException($"Column {columnIndex} is {column.DataType} but requested {typeof(T)}");
            var reader = (IReadOnlyBuffer<T>)column;
            return reader.GetItem(rowIndex);
        }

        public Task<T[]> Get<T>(uint columnIndex, params uint[] rowIndices) where T: notnull
        {
            var column = _columnReader[columnIndex];
            if (column.DataType != typeof(T))
                throw new ArgumentException($"Column {columnIndex} is {column.DataType} but requested {typeof(T)}");
            var reader = (IReadOnlyBuffer<T>)column;
            return reader.GetItems(rowIndices);
        }

        void CreateColumnReader(MethodInfo createColumnReader, BrightDataType dataType, Type type, uint columnIndex, uint offset, uint size)
        {
            var reader = (IReadOnlyBufferWithMetaData)createColumnReader.MakeGenericMethod(type).Invoke(this, [columnIndex, offset, size])!;
            _columnReader[columnIndex] = dataType switch {
                BrightDataType.String            => new MappedReadOnlyBuffer<uint, string>((IReadOnlyBufferWithMetaData<uint>)reader, GetStrings),
                BrightDataType.BinaryData        => new MappedReadOnlyBuffer<DataRangeColumnType, BinaryData>((IReadOnlyBufferWithMetaData<DataRangeColumnType>)reader, GetBinaryData),
                BrightDataType.IndexList         => new MappedReadOnlyBuffer<DataRangeColumnType, IndexList>((IReadOnlyBufferWithMetaData<DataRangeColumnType>)reader, GetIndexLists),
                BrightDataType.WeightedIndexList => new MappedReadOnlyBuffer<DataRangeColumnType, WeightedIndexList>((IReadOnlyBufferWithMetaData<DataRangeColumnType>)reader, GetWeightedIndexLists),
                BrightDataType.Vector            => new MappedReadOnlyBuffer<DataRangeColumnType, ReadOnlyVector>((IReadOnlyBufferWithMetaData<DataRangeColumnType>)reader, _vectorMapper),
                BrightDataType.Matrix            => new MappedReadOnlyBuffer<MatrixColumnType, ReadOnlyMatrix>((IReadOnlyBufferWithMetaData<MatrixColumnType>)reader, _matrixMapper),
                BrightDataType.Tensor3D          => new MappedReadOnlyBuffer<Tensor3DColumnType, ReadOnlyTensor3D>((IReadOnlyBufferWithMetaData<Tensor3DColumnType>)reader, _tensor3DMapper),
                BrightDataType.Tensor4D          => new MappedReadOnlyBuffer<Tensor4DColumnType, ReadOnlyTensor4D>((IReadOnlyBufferWithMetaData<Tensor4DColumnType>)reader, _tensor4DMapper),
                _                                => reader
            };
        }

        public IReadOnlyBufferWithMetaData<T> GetColumn<T>(uint index) where T: notnull
        {
            var typeofT = typeof(T);
            var reader = _columnReader[index];
            var dataType = reader.DataType;

            if(dataType == typeofT)
                return (IReadOnlyBufferWithMetaData<T>)reader;

            if (typeofT == typeof(object)) {
                var ret = new ReadOnlyBufferMetaDataWrapper<object>(GenericTypeMapping.ToObjectConverter(reader), _columnMetaData[index]);
                return (IReadOnlyBufferWithMetaData<T>)ret;
            }

            if (typeofT == typeof(string)) {
                var ret = new ReadOnlyBufferMetaDataWrapper<string>(GenericTypeMapping.ToStringConverter(reader), _columnMetaData[index]);
                return (IReadOnlyBufferWithMetaData<T>)ret;
            }

            if (typeofT.GetTypeInfo().IsAssignableFrom(dataType.GetTypeInfo())) {
                return new ReadOnlyBufferMetaDataWrapper<T>((IReadOnlyBuffer<T>)GenericTypeMapping.CastConverter(typeof(T), reader), _columnMetaData[index]);
            }

            if (dataType.GetBrightDataType().IsNumeric() && typeofT.GetBrightDataType().IsNumeric()) {
                var converter = StaticConverters.GetConverter(dataType, typeof(T));
                return new ReadOnlyBufferMetaDataWrapper<T>((IReadOnlyBuffer<T>)GenericTypeMapping.TypeConverter(typeof(T), reader, converter), _columnMetaData[index]);
            }

            throw new NotImplementedException($"Not able to create a column of type {typeof(T)} from {dataType}");
        }

        public IReadOnlyBufferWithMetaData<TT> GetColumn<FT, TT>(uint index, Func<FT, TT> converter) where FT: notnull where TT : notnull
        {
            var from = GetColumn<FT>(index);
            return (IReadOnlyBufferWithMetaData<TT>)GenericTypeMapping.TypeConverter(typeof(TT), from, new CustomConversionFunction<FT, TT>(converter));
        }

        public IReadOnlyBufferWithMetaData GetColumn(uint index) => _columnReader[index];

        BlockReaderReadOnlyBuffer<T> CreateColumnReader<T>(uint columnIndex, uint offset, uint size) where T : unmanaged => 
            new(_reader, _columnMetaData[columnIndex], offset, size, ReaderBlockSize);

        public GenericTableRow[] Head => EnumerateRows().ToBlockingEnumerable().Take(5).ToArray();
    }
}
