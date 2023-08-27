﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace BrightData.DataTable.Operations
{
    internal class GroupByOperation : OperationBase<(string Label, ICompositeBuffer[] ColumnData)[]>
    {
        readonly BrightDataContext                      _context;
        readonly IProvideTempStreams                    _tempStreams;
        readonly uint[]                                 _groupByColumnIndices;
        readonly MetaData[]                             _columnMetaData;
        readonly ICanEnumerateDisposable[]              _columns;
        readonly IEnumerator<object>[]                  _enumerators;
        readonly Dictionary<string, ICompositeBuffer[]> _groups = new();
        readonly object[]                               _row;

        public GroupByOperation(
            BrightDataContext context,
            IProvideTempStreams tempStreams,
            uint rowCount, 
            uint[] groupByColumnIndices, 
            MetaData[] columnMetaData, 
            ICanEnumerateDisposable[] columns) : base(rowCount, null)
        {
            _context              = context;
            _tempStreams          = tempStreams;
            _groupByColumnIndices = groupByColumnIndices;
            _columnMetaData       = columnMetaData;
            _columns              = columns;
            _enumerators          = columns.Select(r => r.Values.GetEnumerator()).ToArray();
            _row                  = new object[columns.Length];
        }

        public override void Dispose()
        {
            foreach(var item in _enumerators)
                item.Dispose();
            foreach(var item in _columns)
                item.Dispose();
            _tempStreams.Dispose();
        }

        protected override void NextStep(uint index)
        {
            // read a row into the buffer
            for (var i = 0; i < _enumerators.Length; i++) {
                var e = _enumerators[i];
                e.MoveNext();
                _row[i] = e.Current;
            }

            // find the group by row
            var label = GetGroupLabel(_groupByColumnIndices, _row);
            if (!_groups.TryGetValue(label, out var groupBuffers))
                _groups.Add(label, groupBuffers = _columnMetaData.Select(x => x.GetColumnType().GetCompositeBufferWithMetaData(x, _context, _tempStreams)).Cast<ICompositeBuffer>().ToArray());

            // write the row into the group
            foreach(var (obj, buffer) in _row.Zip(groupBuffers))
                buffer.AddObject(obj);
        }

        static string GetGroupLabel(IEnumerable<uint> columnIndices, object[] row) => String.Join('|', 
            columnIndices.Select(ci => row[ci].ToString() ?? throw new Exception("Cannot group by string when value is null"))
        ) ?? throw new Exception("No column indices");

        protected override (string Label, ICompositeBuffer[] ColumnData)[] GetResult(bool wasCancelled)
        {
            return _groups
                .OrderBy(g => g.Key)
                .Select(kv => (kv.Key, kv.Value))
                .ToArray()
            ;
        }
    }
}
