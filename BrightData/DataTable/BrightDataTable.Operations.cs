﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrightData.DataTable.Operations;
using BrightData.Helper;

namespace BrightData.DataTable
{
    public partial class BrightDataTable
    {
        public IOperation<(uint, MetaData)> CreateColumnAnalyser(uint columnIndex, uint writeCount, uint maxDistinctCount)
        {
            var metaData = GetColumnMetaData(columnIndex);
            var type = metaData.GetColumnType();
            var analyser = type.GetColumnAnalyser(metaData, writeCount, maxDistinctCount);
            var reader = GetColumnReader(columnIndex, _header.RowCount);
            var dataType = type.GetDataType();
            var dataType2 = type switch {
                BrightDataType.IndexList => typeof(IHaveIndices),
                BrightDataType.WeightedIndexList => typeof(IHaveIndices),
                BrightDataType.Matrix => typeof(ITensor),
                BrightDataType.Vector => typeof(ITensor),
                BrightDataType.Tensor3D => typeof(ITensor),
                BrightDataType.Tensor4D => typeof(ITensor),
                _ => dataType
            };
            if (dataType != dataType2) {
                return GenericActivator.Create<IOperation<(uint, MetaData)>>(typeof(AnalyseColumnWithCastOperation<,>).MakeGenericType(dataType, dataType2),
                    RowCount,
                    columnIndex,
                    metaData,
                    reader,
                    analyser
                );
            }
            return GenericActivator.Create<IOperation<(uint, MetaData)>>(typeof(AnalyseColumnOperation<>).MakeGenericType(dataType),
                RowCount,
                columnIndex,
                metaData,
                reader,
                analyser
            );
        }

        public IOperation<(string Label, IHybridBuffer[] ColumnData)[]> GroupBy(IProvideTempStreams tempStreams, params uint[] groupByColumnIndices)
        {
            return new GroupByOperation(
                Context,
                tempStreams,
                _header.RowCount,
                groupByColumnIndices,
                ColumnMetaData,
                GetColumnReaders(ColumnIndices)
            );
        }

        public IEnumerable<IOperation<ITypedSegment?>> MutateColumns(IProvideTempStreams temp, IEnumerable<(uint ColumnIndex, IConvertColumn Converter)> converters)
        {
            var converterTable = converters.ToDictionary(d => d.ColumnIndex, d => d.Converter);

            foreach (var ci in ColumnIndices) {
                if (converterTable.TryGetValue(ci, out var converter)) {
                    if (converter.From != ColumnTypes[ci].GetDataType())
                        throw new ArgumentException($"Column types did not agree [{ci}]: Expected {ColumnTypes[ci].GetType()} instead of {converter.From}");

                    var columnReader = GetColumnReader(ci, RowCount);
                    var outputBuffer = converter.To.GetBrightDataType().GetHybridBufferWithMetaData(ColumnMetaData[ci], Context, temp);
                    var operation = GenericActivator.Create<IOperation<ITypedSegment?>>(typeof(ColumnConversionOperation<,>).MakeGenericType(converter.From, converter.To),
                        RowCount,
                        columnReader,
                        converter,
                        outputBuffer
                    );
                    yield return operation;
                }
                else
                    yield return new NopColumnOperation(GetColumn(ci));
            }
        }

        public IEnumerable<IOperation<ITypedSegment?>> ReinterpretColumns(IProvideTempStreams temp, IEnumerable<IReinterpretColumns> columns)
        {
            var rowCount = RowCount;
            var reinterpreted = columns.SelectMany(c => c.SourceColumnIndices.Select(i => (Column: c, Index: i)))
                .ToDictionary(d => d.Index, d => d.Column);

            foreach (var ci in ColumnIndices) {
                if (reinterpreted.TryGetValue(ci, out var rc)) {
                    if (ci == rc.SourceColumnIndices[0]) {
                        foreach (var op in rc.GetNewColumnOperations(Context, temp, rowCount, rc.SourceColumnIndices.Select(i => GetColumnReader(i, rowCount)).ToArray()))
                            yield return op;
                    }
                }
                else
                    yield return new NopColumnOperation(GetColumn(ci));
            }
        }

        public IOperation<BrightDataTableBuilder?> Project(Func<object[], object[]?> projector)
        {
            return new ProjectionOperation(RowCount, Context, GetAllRowData(true).Select(d => d.Data).GetEnumerator(), projector);
        }

        public IOperation<Stream?> BagToStream(uint sampleCount, Stream stream)
        {
            var rowIndices = AllRowIndices.ToArray().Bag(sampleCount, Context.Random);
            return WriteRowsTo(stream, rowIndices);
        }

        public IOperation<Stream?> ShuffleToStream(Stream stream)
        {
            var rowIndices = AllRowIndices.Shuffle(Context.Random).ToArray();
            return WriteRowsTo(stream, rowIndices);
        }
    }
}
