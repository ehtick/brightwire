﻿using System;
using System.IO;
using System.Linq;
using BrightData.LinearAlgebra.ReadOnlyTensorValueSemantics;
using BrightData.LinearAlgebra.Segments;
using CommunityToolkit.HighPerformance;
using CommunityToolkit.HighPerformance.Buffers;

namespace BrightData.LinearAlgebra.ReadOnly
{
    internal class ReadOnlyMatrixWrapper : IReadOnlyMatrix, IEquatable<ReadOnlyMatrixWrapper>, IHaveReadOnlyContiguousFloatSpan
    {
        readonly ReadOnlyMatrixValueSemantics<ReadOnlyMatrixWrapper> _valueSemantics;

        public ReadOnlyMatrixWrapper(IReadOnlyTensorSegment segment, uint rowCount, uint columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            ReadOnlySegment = segment;
            _valueSemantics = new(this);
        }

        public void WriteTo(BinaryWriter writer)
        {
            writer.Write(2);
            writer.Write(ColumnCount);
            writer.Write(RowCount);
            var temp = SpanOwner<float>.Empty;
            ReadOnlySegment.GetSpan(ref temp, out var wasTempUsed);
            try {
                writer.Write(temp.Span.AsBytes());
            }
            finally {
                if (wasTempUsed)
                    temp.Dispose();
            }
        }

        public void Initialize(BrightDataContext context, BinaryReader reader)
        {
            throw new NotImplementedException();
        }

        public ReadOnlySpan<float> FloatSpan => ReadOnlySegment.GetSpan();
        public ReadOnlySpan<float> GetFloatSpan(ref SpanOwner<float> temp, out bool wasTempUsed) => ReadOnlySegment.GetSpan(ref temp, out wasTempUsed);
        public uint RowCount { get; }
        public uint ColumnCount { get; }
        public bool IsReadOnly => true;

        public float this[int rowY, int columnX] => ReadOnlySegment[columnX * RowCount + rowY];
        public float this[uint rowY, uint columnX] => ReadOnlySegment[columnX * RowCount + rowY];

        public IMatrix Create(LinearAlgebraProvider lap) => lap.CreateMatrix(RowCount, ColumnCount, ReadOnlySegment);
        public ReadOnlyTensorSegmentWrapper Row(uint index) => new(ReadOnlySegment, index, RowCount, ColumnCount);
        public ReadOnlyTensorSegmentWrapper Column(uint index) => new(ReadOnlySegment, index * RowCount, 1, RowCount);
        public IReadOnlyVector GetRow(uint rowIndex) => new ReadOnlyVectorWrapper(Row(rowIndex));
        public IReadOnlyVector GetColumn(uint columnIndex) => new ReadOnlyVectorWrapper(Column(columnIndex));
        public IReadOnlyVector[] AllRows() => RowCount.AsRange().Select(GetRow).ToArray();
        public IReadOnlyVector[] AllColumns() => ColumnCount.AsRange().Select(GetColumn).ToArray();
        public uint Size => RowCount * ColumnCount;
        public IReadOnlyTensorSegment ReadOnlySegment { get; }

        // value semantics
        public override bool Equals(object? obj) => _valueSemantics.Equals(obj as ReadOnlyMatrixWrapper);
        public override int GetHashCode() => _valueSemantics.GetHashCode();
        public bool Equals(ReadOnlyMatrixWrapper? other) => _valueSemantics.Equals(other);

        public override string ToString()
        {
            var preview = String.Join("|", ReadOnlySegment.Values.Take(Consts.DefaultPreviewSize));
            if (Size > Consts.DefaultPreviewSize)
                preview += "|...";
            return $"Read Only Matrix Wrapper (Rows: {RowCount}, Columns: {ColumnCount}) {preview}";
        }
    }
}
