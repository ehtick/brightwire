﻿using System;
using System.Collections.Generic;
using System.Linq;

namespace BrightData.DataTable.Operations
{
    internal class ManyToOneColumnOperation<T> : OperationBase<ITableSegment?>
        where T: notnull
    {
        readonly ICanEnumerateDisposable[] _input;
        readonly IEnumerator<object>[] _enumerators;
        readonly Func<object[], T> _converter;
        readonly object[] _temp;
        readonly ICompositeBuffer<T> _outputBuffer;

        public ManyToOneColumnOperation(
            uint rowCount,
            ICanEnumerateDisposable[] input, 
            Func<object[], T> converter, 
            ICompositeBuffer<T> outputBuffer) : base(rowCount, null)
        {
            _input = input;
            _converter = converter;
            _outputBuffer = outputBuffer;
            _enumerators = input.Select(x => x.Values.GetEnumerator()).ToArray();
            _temp = new object[input.Length];
        }

        public override void Dispose()
        {
            foreach(var item in _enumerators)
                item.Dispose();
            foreach(var item in _input)
                item.Dispose();
        }

        protected override void NextStep(uint index)
        {
            for (var i = 0; i < _enumerators.Length; i++) {
                var e = _enumerators[i];
                e.MoveNext();
                _temp[i] = e.Current;
            }

            _outputBuffer.Add(_converter(_temp));
        }

        protected override ITableSegment? GetResult(bool wasCancelled)
        {
            if (wasCancelled)
                return null;

            return (ITableSegment) _outputBuffer;
        }
    }
}