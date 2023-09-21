﻿namespace BrightData.Buffer.ReadOnly.Converter
{
    internal class ToObjectConverter<FT> : ReadOnlyConverterBase<FT, object> where FT: notnull
    {
        public ToObjectConverter(IReadOnlyBuffer<FT> from) : base(from)
        {
        }

        protected override void Convert(in FT from, ref object to)
        {
            to = from;
        }

        protected override object Convert(in FT from)
        {
            return from;
        }
    }
}
