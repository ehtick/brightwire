﻿using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrightData.Buffer;

namespace BrightData.DataTable.Consumers
{
    /// <summary>
    /// A typed data table segment that can grow in size
    /// </summary>
    /// <typeparam name="T"></typeparam>
    internal class GrowableDataTableSegment<T> : IDataTableSegment<T>, IConsumeColumnData<T>
        where T: notnull
    {
        readonly GrowableSegment<T> _segment;

        public GrowableDataTableSegment(IBrightDataContext context, IColumnInfo column, uint size, IProvideTempStreams tempStream)
        {
            Context = context;
            SingleType = column.ColumnType;
            ColumnIndex = column.Index;
            Types = new[] { SingleType };
            Size = size;
            var buffer = column.MetaData.GetGrowableSegment(column.ColumnType, context, tempStream);

            MetaData = column.MetaData;
            _segment = new GrowableSegment<T>(SingleType, MetaData, (IHybridBuffer<T>)buffer);
        }

        public GrowableDataTableSegment(IBrightDataContext context, IColumnInfo column, uint size, IProvideTempStreams tempStream, IEnumerable<T> data) : this(context, column, size, tempStream)
        {
            uint index = 0;
            foreach (var item in data)
                _segment.Add(item, index++);
        }

        public void Dispose() => _segment.Dispose();

        public void Add(object value, uint index) => Add((T) value, index);
        public void Add(T value, uint index) => _segment.Add(value, index);
        public IBrightDataContext Context { get; }
        public IEnumerable<T> EnumerateTyped() => _segment.EnumerateTyped();
        public IEnumerable<object> Enumerate() => EnumerateTyped().Cast<object>();
        public BrightDataType[] Types { get; }
        public uint Size { get; }
        public IEnumerable<object?> Data => _segment.Enumerate();
        public IMetaData MetaData { get; }
        public BrightDataType SingleType { get; }
        public uint ColumnIndex { get; }
        public BrightDataType ColumnType => SingleType;
        public void WriteTo(BinaryWriter writer) => _segment.WriteTo(writer);
    }
}
