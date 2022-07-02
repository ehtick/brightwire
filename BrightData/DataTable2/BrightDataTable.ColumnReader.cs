﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrightData.DataTable2.TensorData;

namespace BrightData.DataTable2
{
    public partial class BrightDataTable
    {
        static void ValidateColumnTypes(Type columnType, Type requestedType)
        {
            if (columnType == typeof(IVector)) {
                if(requestedType != typeof(VectorData))
                    throw new ArgumentException($"Use {nameof(VectorData)} instead");
            }
            else if (columnType == typeof(IMatrix)) {
                if(requestedType != typeof(MatrixData))
                    throw new ArgumentException($"Use {nameof(MatrixData)} instead");
            }
            else if (columnType == typeof(ITensor3D)) {
                if(requestedType != typeof(Tensor3DData))
                    throw new ArgumentException($"Use {nameof(Tensor3DData)} instead");
            }
            else if (columnType == typeof(ITensor4D)) {
                if(requestedType != typeof(Tensor4DData))
                    throw new ArgumentException($"Use {nameof(Tensor4DData)} instead");
            }
            else if (columnType != requestedType)
                throw new ArgumentException($"Data types do not align - expected {columnType} but received {requestedType}");
        }

        ICanEnumerateDisposable<T> GetColumnReader<T>(uint columnIndex, uint countToRead, Func<uint, uint>? offsetAdjuster = null) where T : notnull
        {
            ref readonly var column = ref _columns[columnIndex];
            var requestedType = typeof(T);
            ValidateColumnTypes(column.DataType.GetDataType(), requestedType);

            var offset = _columnOffset[columnIndex];
            if(offsetAdjuster is not null)
                offset += offsetAdjuster(column.DataTypeSize);
            var (columnDataType, _) = column.DataType.GetColumnType();
            var sizeInBytes = countToRead * column.DataTypeSize;
            return (ICanEnumerateDisposable<T>)_getReader.MakeGenericMethod(columnDataType, requestedType).Invoke(this, new object[] { offset, sizeInBytes })!;
        }

        ICanEnumerateDisposable GetColumnReader(uint columnIndex, uint countToRead, Func<uint, uint>? offsetAdjuster = null)
        {
            ref readonly var column = ref _columns[columnIndex];
            var dataType = column.DataType.GetDataType();
            if (dataType == typeof(IVector))
                dataType = typeof(VectorData);
            else if(dataType == typeof(IMatrix))
                dataType = typeof(MatrixData);

            var offset = _columnOffset[columnIndex];
            if(offsetAdjuster is not null)
                offset += offsetAdjuster(column.DataTypeSize);
            var (columnDataType, _) = column.DataType.GetColumnType();
            var sizeInBytes = countToRead * column.DataTypeSize;
            return (ICanEnumerateDisposable)_getReader.MakeGenericMethod(columnDataType, dataType).Invoke(this, new object[] { offset, sizeInBytes })!;
        }

        ICanEnumerateDisposable[] GetColumnReaders(IEnumerable<uint> columnIndices)
        {
            var columnCount = _header.ColumnCount;
            var ret = new ICanEnumerateDisposable[columnCount];
            for (uint i = 0; i < columnCount; i++)
                ret[i] = GetColumnReader(i, _header.RowCount);
            return ret;
        }

        ICanEnumerate<T> GetReader<CT, T>(uint offset, uint sizeInBytes)
            where T: notnull
            where CT: unmanaged
        {
            IConvertStructsToObjects<CT, T> GetConverter()
            {
                var dataType = typeof(T);

                if (dataType == typeof(string)) {
                    var ret = new StringColumnConverter(_stringTable.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                if (dataType == typeof(IndexList)) {
                    var ret = new IndexListConverter(_indices.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                if (dataType == typeof(WeightedIndexList)) {
                    var ret = new WeightedIndexListConverter(_weightedIndices.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                if (dataType == typeof(IVectorInfo)) {
                    var ret = new VectorInfoConverter(Context, _tensors.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                if (dataType == typeof(IMatrixInfo)) {
                    var ret = new MatrixInfoConverter(Context, _tensors.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                if (dataType == typeof(ITensor3DInfo)) {
                    var ret = new Tensor3DInfoConverter(Context, _tensors.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                if (dataType == typeof(ITensor4DInfo)) {
                    var ret = new Tensor4DInfoConverter(Context, _tensors.Value);
                    return (IConvertStructsToObjects<CT, T>)ret;
                }

                return new NopColumnConverter<CT, T>();
            }

            var block = _buffer.GetIterator<CT>(offset, sizeInBytes);
            var converter = GetConverter();
            return new ColumnReader<CT, T>(block.GetEnumerator(), converter, block);
        }

        class NopColumnConverter<CT, T> : IConvertStructsToObjects<CT, T>
            where CT : unmanaged
            where T: notnull
        {
            public T Convert(ref CT item)
            {
                return __refvalue(__makeref(item), T);
            }
        }

        class StringColumnConverter : IConvertStructsToObjects<uint, string>
        {
            readonly string[] _stringTable;

            public StringColumnConverter(string[] stringTable)
            {
                _stringTable = stringTable;
            }

            public string Convert(ref uint item) => _stringTable[item];
        }

        class IndexListConverter : IConvertStructsToObjects<DataRangeColumnType, IndexList>
        {
            readonly ICanRandomlyAccessData<uint> _indices;

            public IndexListConverter(ICanRandomlyAccessData<uint> indices)
            {
                _indices = indices;
            }

            public IndexList Convert(ref DataRangeColumnType item)
            {
                var span = _indices.GetSpan(item.StartIndex, item.Count);
                return IndexList.Create(span);
            }
        }

        class WeightedIndexListConverter : IConvertStructsToObjects<DataRangeColumnType, WeightedIndexList>
        {
            readonly ICanRandomlyAccessData<WeightedIndexList.Item> _indices;

            public WeightedIndexListConverter(ICanRandomlyAccessData<WeightedIndexList.Item> indices)
            {
                _indices = indices;
            }

            public WeightedIndexList Convert(ref DataRangeColumnType item)
            {
                var span = _indices.GetSpan(item.StartIndex, item.Count);
                return WeightedIndexList.Create(span);
            }
        }

        class VectorInfoConverter : IConvertStructsToObjects<DataRangeColumnType, VectorData>
        {
            readonly BrightDataContext _context;
            readonly ICanRandomlyAccessData<float> _data;

            public VectorInfoConverter(BrightDataContext context, ICanRandomlyAccessData<float> data)
            {
                _context = context;
                _data = data;
            }

            public VectorData Convert(ref DataRangeColumnType item)
            {
                return new VectorData(_context, _data, item.StartIndex, item.Count);
            }
        }

        class MatrixInfoConverter : IConvertStructsToObjects<MatrixColumnType, MatrixData>
        {
            readonly BrightDataContext _context;
            readonly ICanRandomlyAccessData<float> _data;

            public MatrixInfoConverter(BrightDataContext context, ICanRandomlyAccessData<float> data)
            {
                _context = context;
                _data = data;
            }

            public MatrixData Convert(ref MatrixColumnType item)
            {
                return new MatrixData(_context, _data, item.StartIndex, item.RowCount, item.ColumnCount);
            }
        }

        class Tensor3DInfoConverter : IConvertStructsToObjects<Tensor3DColumnType, Tensor3DData>
        {
            readonly BrightDataContext _context;
            readonly ICanRandomlyAccessData<float> _data;

            public Tensor3DInfoConverter(BrightDataContext context, ICanRandomlyAccessData<float> data)
            {
                _context = context;
                _data = data;
            }

            public Tensor3DData Convert(ref Tensor3DColumnType item)
            {
                return new Tensor3DData(_context, _data, item.StartIndex, item.Depth, item.RowCount, item.ColumnCount);
            }
        }

        class Tensor4DInfoConverter : IConvertStructsToObjects<Tensor4DColumnType, Tensor4DData>
        {
            readonly BrightDataContext _context;
            readonly ICanRandomlyAccessData<float> _data;

            public Tensor4DInfoConverter(BrightDataContext context, ICanRandomlyAccessData<float> data)
            {
                _context = context;
                _data = data;
            }

            public Tensor4DData Convert(ref Tensor4DColumnType item)
            {
                return new Tensor4DData(_context, _data, item.StartIndex, item.Count, item.Depth, item.RowCount, item.ColumnCount);
            }
        }
    }
}
