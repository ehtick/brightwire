﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Microsoft.Toolkit.HighPerformance.Buffers;

namespace BrightData.LinearAlegbra2.TensorData
{
    internal class MatrixData : IMatrixInfo
    {
        readonly TensorSegmentWrapper2 _segment;

        public MatrixData(TensorSegmentWrapper2 segment, uint rowCount, uint columnCount)
        {
            RowCount = rowCount;
            ColumnCount = columnCount;
            _segment = segment;
        }

        public void WriteTo(BinaryWriter writer)
        {
            writer.Write(2);
            writer.Write(ColumnCount);
            writer.Write(RowCount);
            var temp = SpanOwner<float>.Empty;
            _segment.GetSpan(ref temp, out var wasTempUsed);
            try {
                writer.Write(temp.Span.AsBytes());
            }
            finally {
                if(wasTempUsed)
                    temp.Dispose();
            }
        }

        public void Initialize(BrightDataContext context, BinaryReader reader)
        {
            throw new NotImplementedException();
        }

        public ReadOnlySpan<float> GetSpan(ref SpanOwner<float> temp, out bool wasTempUsed) => _segment.GetSpan(ref temp, out wasTempUsed);
        public uint RowCount { get; }
        public uint ColumnCount { get; }

        public float this[int rowY, int columnX] => _segment[columnX * RowCount + rowY];
        public float this[uint rowY, uint columnX] => _segment[columnX * RowCount + rowY];

        public IMatrix Create(LinearAlgebraProvider lap) => lap.CreateMatrix(RowCount, ColumnCount, _segment);
        public IVectorInfo GetRow(uint rowIndex) => new VectorData(new TensorSegmentWrapper2(_segment, rowIndex, RowCount, ColumnCount));
        public IVectorInfo GetColumn(uint columnIndex) => new VectorData(new TensorSegmentWrapper2(_segment, columnIndex * RowCount, 1, RowCount));
        public uint Size => RowCount * ColumnCount;
    }
}
