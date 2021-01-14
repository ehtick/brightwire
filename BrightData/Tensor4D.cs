﻿using System.IO;
using BrightData.Memory;

namespace BrightData
{
    public class Tensor4D<T> : TensorBase<T, Tensor4D<T>>
        where T: struct
    {
        internal Tensor4D(ITensorSegment<T> segment, uint count, uint depth, uint rows, uint columns) : base(segment, new[] { count, depth, rows, columns }) { }
        internal Tensor4D(IBrightDataContext context, BinaryReader reader) : base(context, reader) { }

        public uint Count => Shape[0];
        public uint Depth => Shape[1];
        public uint RowCount => Shape[2];
        public uint ColumnCount => Shape[3];
        public uint MatrixSize => RowCount * ColumnCount;
        public uint TensorSize => Depth * MatrixSize;
        public new uint Size => Count * TensorSize;

        public T this[int count, int depth, int rowY, int columnX]
        {
            get => _segment[count * TensorSize + depth * MatrixSize + rowY * ColumnCount + columnX];
            set => _segment[count * TensorSize + depth * MatrixSize + rowY * ColumnCount + columnX] = value;
        }
        public T this[uint count, uint depth, uint rowY, uint columnX]
        {
            get => _segment[count * TensorSize + depth * MatrixSize + rowY * ColumnCount + columnX];
            set => _segment[count * TensorSize + depth * MatrixSize + rowY * ColumnCount + columnX] = value;
        }

        public Tensor3D<T> Tensor(uint index)
        {
            var segment = new TensorSegmentWrapper<T>(_segment, index * TensorSize, 1, TensorSize);
            return new Tensor3D<T>(segment, Depth, RowCount, ColumnCount);
        }

        public override string ToString() => $" Tensor4D (Count: {Count}, Depth: {Depth}, Rows: {RowCount}, Columns: {ColumnCount})";

        protected override Tensor4D<T> Create(ITensorSegment<T> segment)
        {
            return new Tensor4D<T>(segment, Count, Depth, RowCount, ColumnCount);
        }
    }
}
