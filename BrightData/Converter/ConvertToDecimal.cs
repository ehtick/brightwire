﻿using System;

namespace BrightData.Converter
{
    internal class ConvertToDecimal<T> : ConverterBase<T>, ICanConvert<T, decimal>
        where T : notnull
    {
        readonly Func<T, decimal> _converter;

        public ConvertToDecimal(bool throwOnFailure = false) : base(throwOnFailure)
        {
            var typeCode = Type.GetTypeCode(typeof(T));
            _converter = typeCode switch
            {
                TypeCode.String  => FromString,
                TypeCode.Single  => FromSingle,
                TypeCode.Double  => FromDouble,
                TypeCode.SByte   => FromSByte,
                TypeCode.Byte    => FromByte,
                TypeCode.Int16   => FromInt16,
                TypeCode.UInt16  => FromUInt16,
                TypeCode.Int32   => FromInt32,
                TypeCode.UInt32  => FromUInt32,
                TypeCode.Int64   => FromInt64,
                TypeCode.UInt64  => FromUInt64,
                TypeCode.Decimal => GetDecimal,
                _                => ConvertGeneric,
            };
        }

        static decimal FromString(T str) => decimal.Parse(__refvalue(__makeref(str), string));
        decimal FromSingle(T data)       => System.Convert.ToDecimal(GetSingle(data));
        decimal FromDouble(T data)       => System.Convert.ToDecimal(GetDouble(data));
        decimal FromSByte(T data)        => GetSByte(data);
        decimal FromByte(T data)         => GetByte(data);
        decimal FromInt16(T data)        => GetInt16(data);
        decimal FromUInt16(T data)       => GetUInt16(data);
        decimal FromInt32(T data)        => GetInt32(data);
        decimal FromUInt32(T data)       => GetUInt32(data);
        decimal FromInt64(T data)        => GetInt64(data);
        decimal FromUInt64(T data)       => GetUInt64(data);
        decimal ConvertGeneric(T data)
        {
            var (ret, wasConverted) = _genericConverter.Value.ConvertValue(data);
            if(!wasConverted && _throwOnFailure || ret == null)
                throw new ArgumentException($"Could not convert {data} to decimal");
            return (decimal)ret;
        }

        public decimal Convert(T data) => _converter(data);
        public Type To => typeof(decimal);
    }
}
