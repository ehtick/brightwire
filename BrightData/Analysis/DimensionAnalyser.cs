﻿using System;
using System.Collections.Generic;
using BrightData.LinearAlgebra;

namespace BrightData.Analysis
{
    internal class DimensionAnalyser : IDataAnalyser<ITensor2>
    {
        readonly uint _maxCount;
        readonly HashSet<(uint X, uint Y, uint Z)> _distinct = new();

        public DimensionAnalyser(uint maxCount = Consts.MaxDistinct)
        {
            _maxCount = maxCount;
        }

        public uint? XDimension { get; private set; }
        public uint? YDimension { get; private set; }
        public uint? ZDimension { get; private set; }

        public void Add(ITensor2 obj)
        {
            uint x = 0, y = 0, z = 0;
            if (obj is IVector vector) {
                if (XDimension == null || vector.Size > XDimension)
                    XDimension = x = vector.Size;
            } else if (obj is IMatrix matrix) {
                if (XDimension == null || matrix.ColumnCount > XDimension)
                    XDimension = x = matrix.ColumnCount;
                if (YDimension == null || matrix.RowCount > YDimension)
                    YDimension = y = matrix.RowCount;
            } else if (obj is ITensor3D tensor) {
                if (XDimension == null || tensor.ColumnCount > XDimension)
                    XDimension = x = tensor.ColumnCount;
                if (YDimension == null || tensor.RowCount > YDimension)
                    YDimension = y = tensor.RowCount;
                if (ZDimension == null || tensor.Depth > ZDimension)
                    ZDimension = z = tensor.Depth;
            } else
                throw new NotImplementedException();

            if (_distinct.Count < _maxCount)
                _distinct.Add((x, y, z));
        }

        public void AddObject(object obj)
        {
            if (obj is ITensor2 tensor)
                Add(tensor);
            else
                throw new ArgumentException("Expected a tensor", nameof(obj));
        }

        public void WriteTo(IMetaData metadata)
        {
            metadata.Set(Consts.HasBeenAnalysed, true);
            metadata.SetIfNotNull(Consts.XDimension, XDimension);
            metadata.SetIfNotNull(Consts.YDimension, YDimension);
            metadata.SetIfNotNull(Consts.ZDimension, ZDimension);
            if (_distinct.Count < _maxCount)
                metadata.Set(Consts.NumDistinct, (uint)_distinct.Count);

            var size = (XDimension ?? 1) * (YDimension ?? 1) * (ZDimension ?? 1);
            metadata.Set(Consts.Size, size);
        }
    }
}
