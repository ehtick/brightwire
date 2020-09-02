﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BrightData
{
    public class Vector<T> : TensorBase<T, Vector<T>>
        where T: struct
    {
        public Vector(ITensorSegment<T> data) : base(data, new[] { data.Size }) { }
        public Vector(IBrightDataContext context, BinaryReader reader) : base(context, reader) { }

        public new uint Size => Shape[0];

        protected override Vector<T> Create(ITensorSegment<T> segment) => new Vector<T>(segment);

        public T this[int index]
        {
            get => _data[(uint)index];
            set => _data[(uint)index] = value;
        }

        public T this[uint index]
        {
            get => _data[index];
            set => _data[index] = value;
        }

        public IEnumerable<T> Values => _data.Values;

        public override string ToString()
        {
            var preview = String.Join("|", _data.Values.Take(8));
            if (Size > 8)
                preview += "|...";
            return $"Vector ({Size}): {preview}";
        }

        public void CopyFrom(T[] array) => _data.Initialize(array);
    }
}
