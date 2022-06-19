﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace BrightData.DataTable2
{
    public partial class BrightDataTable
    {
        enum TypeConversion
        {
            Cast,
            ToString,
            DateTicks,
            ChangeType
        }
        readonly Lazy<Dictionary<(uint Index, Type TargetType), TypeConversion>> _typeConversionTable = new();

        public T ConvertObjectTo<T>(uint index, object ret) where T: notnull
        {
            var targetType = typeof(T);
            var key = (index, targetType);
            if (!_typeConversionTable.Value.TryGetValue(key, out var typeConversion)) {
                var retType = ret.GetType();
                if (retType == targetType || targetType.GetTypeInfo().IsAssignableFrom(retType.GetTypeInfo()))
                    typeConversion = TypeConversion.Cast;
                else if (retType == typeof(DateTime))
                    typeConversion = targetType == typeof(string) ? TypeConversion.ToString : TypeConversion.DateTicks;
                else
                    typeConversion = TypeConversion.ChangeType;
                _typeConversionTable.Value.Add(key, typeConversion);
            }

            return typeConversion switch {
                TypeConversion.Cast => (T) ret,
                TypeConversion.ToString => (T) (object) (ret.ToString() ?? ""),
                TypeConversion.DateTicks => (T) Convert.ChangeType(((DateTime) ret).Ticks, targetType),
                TypeConversion.ChangeType => (T) Convert.ChangeType(ret, targetType),
                _ => throw new NotImplementedException()
            };
        }
    }
}
