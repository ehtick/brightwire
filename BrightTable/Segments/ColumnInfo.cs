﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using BrightData;

namespace BrightTable.Segments
{
    class ColumnInfo : IColumnInfo
    {
        public ColumnInfo(BinaryReader reader, uint index)
        {
            ColumnType = (ColumnType)reader.ReadSByte();
            MetaData = new MetaData(reader);
            MetaData.Set(Consts.Index, index);
            Index = index;
        }

        public ColumnInfo(uint index, ColumnType type, IMetaData metaData)
        {
            Index = index;
            ColumnType = type;
            MetaData = metaData;
            MetaData.Set(Consts.Index, index);
        }

        public uint Index { get; }
        public ColumnType ColumnType { get; }
        public IMetaData MetaData { get; }

        public void WriteTo(BinaryWriter writer)
        {
            writer.Write((sbyte) ColumnType);
            MetaData.WriteTo(writer);
        }

        public override string ToString()
        {
            return $"[{ColumnType}]: {MetaData}";
        }
    }
}
