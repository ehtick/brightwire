﻿using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrightData.Analysis;
using BrightData.Buffer.Composite;
using BrightData.Buffer.ReadOnly.Converter;
using BrightData.Helper;
using BrightData.LinearAlgebra.ReadOnly;
using BrightData.Operations.Conversion;
using BrightData.Operations.Helper;
using BrightData.Operations;
using CommunityToolkit.HighPerformance.Buffers;

namespace BrightData
{
    public partial class ExtensionMethods
    {
        /// <summary>
        /// Copies all values from a tensor segment into a float buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="segment"></param>
        public static void CopyFrom(this ICompositeBuffer<float> buffer, INumericSegment<float> segment)
        {
            for(uint i = 0, len = segment.Size; i < len; i++)
                buffer.Add(segment[i]);
        }

        /// <summary>
        /// Copies all values from a span into a float buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="span"></param>
        public static void CopyFrom(this ICompositeBuffer<float> buffer, ReadOnlySpan<float> span)
        {
            for(int i = 0, len = span.Length; i < len; i++)
                buffer.Add(span[i]);
        }

        public static IEnumerable<object> GetValues(this IReadOnlyBuffer buffer)
        {
            return buffer.EnumerateAll().ToBlockingEnumerable();
        }

        public static IAsyncEnumerable<T> GetValues<T>(this IReadOnlyBuffer buffer) where T: notnull
        {
            if (buffer.DataType != typeof(T))
                throw new ArgumentException($"Buffer is of type {buffer.DataType} but requested {typeof(T)}");
            var typedBuffer = (IReadOnlyBuffer<T>)buffer;
            return typedBuffer.EnumerateAllTyped();
        }

        public static IReadOnlyBuffer<string> ToReadOnlyStringBuffer(this IReadOnlyBuffer buffer)
        {
            if (buffer.DataType == typeof(string))
                return (IReadOnlyBuffer<string>)buffer;
            return GenericActivator.Create<IReadOnlyBuffer<string>>(typeof(ToStringConverter<>).MakeGenericType(typeof(string)), buffer);
        }

        public static IReadOnlyBuffer<TT> Convert<FT, TT>(this IReadOnlyBuffer<FT> buffer, Func<FT, TT> converter)
            where FT: notnull
            where TT: notnull
        {
            return GenericActivator.Create<IReadOnlyBuffer<TT>>(typeof(CustomConverter<,>).MakeGenericType(typeof(FT), typeof(TT)), buffer, converter);
        }

        public static async Task<Dictionary<string, List<uint>>> GetGroups(this IReadOnlyBuffer[] buffers)
        {
            var enumerators = buffers.Select(x => x.EnumerateAll().GetAsyncEnumerator()).ToArray();
            var shouldContinue = true;
            var sb = new StringBuilder();
            Dictionary<string, List<uint>> ret = new();
            uint rowIndex = 0;

            while (shouldContinue) {
                sb.Clear();
                foreach (var enumerator in enumerators) {
                    if (!await enumerator.MoveNextAsync()) {
                        shouldContinue = false; 
                        break;
                    }
                    if (sb.Length > 0)
                        sb.Append('|');
                    sb.Append(enumerator.Current);
                }
                var str = sb.ToString();
                if(!ret.TryGetValue(str, out var list))
                    ret.Add(str, list = new());
                list.Add(rowIndex++);
            }

            return ret;
        }

        class MemorySegment<T> : ReadOnlySequenceSegment<T>
        {
            public MemorySegment(ReadOnlyMemory<T> memory) => Memory = memory;
            public MemorySegment<T> Append(ReadOnlyMemory<T> memory)
            {
                var segment = new MemorySegment<T>(memory) {
                    RunningIndex = RunningIndex + Memory.Length
                };
                Next = segment;
                return segment;
            }
        }

        public static async Task<ReadOnlySequence<T>> AsReadOnlySequence<T>(this IReadOnlyBuffer<T> buffer) where T : notnull
        {
            if(buffer.BlockCount == 0)
                return ReadOnlySequence<T>.Empty;

            var first = new MemorySegment<T>(await buffer.GetTypedBlock(0));
            var last = first;
            for(var i = 1; i < buffer.BlockCount; i++)
                last = last.Append(await buffer.GetTypedBlock(1));
            return new ReadOnlySequence<T>(first, 0, last, last.Memory.Length);
        }

        public static async Task<T> GetItem<T>(this IReadOnlyBuffer<T> buffer, uint index) where T: notnull
        {
            var blockIndex = index / buffer.BlockSize;
            var blockMemory = await buffer.GetTypedBlock(blockIndex);
            var ret = blockMemory.Span[(int)(index % buffer.BlockSize)];
            return ret;
        }

        public static async Task<T[]> GetItems<T>(this IReadOnlyBuffer<T> buffer, uint[] indices) where T: notnull
        {
            var blocks = indices.Select(x => (Index: x, BlockIndex: x / buffer.BlockSize, RelativeIndex: x % buffer.BlockSize))
                    .GroupBy(x => x.BlockIndex)
                    .OrderBy(x => x.Key)
                ;
            var ret = new T[indices.Length];
            foreach (var block in blocks) {
                var blockMemory = await buffer.GetTypedBlock(block.Key);
                Add(blockMemory, block, ret);
            }
            return ret;

            static void Add(ReadOnlyMemory<T> data, IEnumerable<(uint Index, uint BlockIndex, uint RelativeIndex)> list, T[] output)
            {
                var span = data.Span;
                foreach (var (index, _, relativeIndex) in list)
                    output[relativeIndex] = span[(int)index];
            }
        }

        public ref struct ReadOnlyBufferIterator<T> where T: notnull
        {
            readonly IReadOnlyBuffer<T> _buffer;
            ReadOnlyMemory<T> _currentBlock = ReadOnlyMemory<T>.Empty;
            uint _blockIndex = 0, _position = 0;

            public ReadOnlyBufferIterator(IReadOnlyBuffer<T> buffer) => _buffer = buffer;

            public bool MoveNext()
            {
                if (++_position < _currentBlock.Length)
                    return true;

                while(_blockIndex < _buffer.BlockCount) {
                    _currentBlock = _buffer.GetTypedBlock(_blockIndex++).Result;
                    if (_currentBlock.Length > 0) {
                        _position = 0;
                        return true;
                    }
                }
                return false;
            }

            public readonly ref readonly T Current => ref _currentBlock.Span[(int)_position];
            public readonly ReadOnlyBufferIterator<T> GetEnumerator() => this;
        }
        public static ReadOnlyBufferIterator<T> GetEnumerator<T>(this IReadOnlyBuffer<T> buffer) where T: notnull => new(buffer);

        public static async Task<T[]> ToArray<T>(this IReadOnlyBuffer<T> buffer) where T : notnull
        {
            var ret = new T[buffer.Size];
            var offset = 0;
            await buffer.ForEachBlock(x => {
                x.CopyTo(new Span<T>(ret, offset, x.Length));
                offset += x.Length;
            });
            return ret;
        }

        public static ICompositeBuffer<string> CreateCompositeBuffer(
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) => new StringCompositeBuffer(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        public static ICompositeBuffer<T> CreateCompositeBuffer<T>(
            CreateFromReadOnlyByteSpan<T> createItem,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) where T: IHaveDataAsReadOnlyByteSpan => new ManagedCompositeBuffer<T>(createItem, tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        public static ICompositeBuffer<T> CreateCompositeBuffer<T>(
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) where T: unmanaged => new UnmanagedCompositeBuffer<T>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        public static ICompositeBuffer CreateCompositeBuffer(
            this BrightDataType dataType,
            IProvideTempData? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null)
        {
            return dataType switch {
                BrightDataType.BinaryData        => CreateCompositeBuffer<BinaryData>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Boolean           => CreateCompositeBuffer<bool>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Date              => CreateCompositeBuffer<DateTime>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.DateOnly          => CreateCompositeBuffer<DateOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Decimal           => CreateCompositeBuffer<DateOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.SByte             => CreateCompositeBuffer<byte>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Short             => CreateCompositeBuffer<short>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Int               => CreateCompositeBuffer<int>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Long              => CreateCompositeBuffer<long>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Float             => CreateCompositeBuffer<float>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Double            => CreateCompositeBuffer<double>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.String            => CreateCompositeBuffer(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.IndexList         => CreateCompositeBuffer<IndexList>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.WeightedIndexList => CreateCompositeBuffer<WeightedIndexList>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Vector            => CreateCompositeBuffer<ReadOnlyVector>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Matrix            => CreateCompositeBuffer<ReadOnlyMatrix>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Tensor3D          => CreateCompositeBuffer<ReadOnlyTensor3D>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Tensor4D          => CreateCompositeBuffer<ReadOnlyTensor4D>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.TimeOnly          => CreateCompositeBuffer<TimeOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                _                                => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, $"Not able to create a composite buffer for type: {dataType}")
            };
        }

        public static IBufferWriter<T> AsBufferWriter<T>(this ICompositeBuffer<T> buffer, int bufferSize = 256) where T : notnull => new CompositeBufferWriter<T>(buffer, bufferSize);

        public static bool CanEncode<T>(this ICompositeBuffer<T> buffer) where T : notnull => buffer.DistinctItems.HasValue;

        /// <summary>
        /// Encoding a composite buffer maps each item to an index and returns both the mapping table and a new composite buffer of the indices
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static (T[] Table, ICompositeBuffer<uint> Data) Encode<T>(
            this ICompositeBuffer<T> buffer, 
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null
        ) where T : notnull {
            if(!buffer.DistinctItems.HasValue)
                throw new ArgumentException("Buffer cannot be encoded as the number of distinct items is not known - create the composite buffer with a high max distinct items", nameof(buffer));
            var table = new Dictionary<T, uint>((int)buffer.DistinctItems.Value);
            var data = new UnmanagedCompositeBuffer<uint>(tempStreams, blockSize, maxInMemoryBlocks);

            buffer.ForEachBlock(block => {
                var len = block.Length;
                if (len == 1) {
                    var item = block[0];
                    if(!table.TryGetValue(item, out var index))
                        table.Add(item, index = (uint)table.Count);
                    data.Add(index);
                }
                else if (len > 1) {
                    var spanOwner = SpanOwner<uint>.Empty;
                    var indices = len <= Consts.MaxStackAllocSize / sizeof(uint)
                        ? stackalloc uint[len]
                        : (spanOwner = SpanOwner<uint>.Allocate(len)).Span
                    ;
                    try {
                        // encode the block
                        for(var i = 0; i < len; i++) {
                            ref readonly var item = ref block[i];
                            if(!table.TryGetValue(item, out var index))
                                table.Add(item, index = (uint)table.Count);
                            indices[i] = index;
                        }
                        data.Add(indices);
                    }
                    finally {
                        if (spanOwner.Length > 0)
                            spanOwner.Dispose();
                    }
                }
            });

            var ret = new T[table.Count];
            foreach (var item in table)
                ret[item.Value] = item.Key;
            return (ret, data);
        }

        public static ICompositeBuffer GetCompositeBuffer(this Type type,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) => GetCompositeBuffer(GetTableDataType(type), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        public static ICompositeBuffer GetCompositeBuffer(this BrightDataType type,
            IProvideTempData? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) => type switch 
        {
            BrightDataType.BinaryData        => CreateCompositeBuffer<BinaryData>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Boolean           => CreateCompositeBuffer<bool>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.SByte             => CreateCompositeBuffer<sbyte>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Short             => CreateCompositeBuffer<short>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Int               => CreateCompositeBuffer<int>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Long              => CreateCompositeBuffer<long>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Float             => CreateCompositeBuffer<float>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Double            => CreateCompositeBuffer<double>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Decimal           => CreateCompositeBuffer<decimal>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.String            => CreateCompositeBuffer<uint>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Date              => CreateCompositeBuffer<DateTime>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.IndexList         => CreateCompositeBuffer<IndexList>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.WeightedIndexList => CreateCompositeBuffer<WeightedIndexList>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Vector            => CreateCompositeBuffer<ReadOnlyVector>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Matrix            => CreateCompositeBuffer<ReadOnlyMatrix>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Tensor3D          => CreateCompositeBuffer<ReadOnlyTensor3D>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.Tensor4D          => CreateCompositeBuffer<ReadOnlyTensor4D>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.TimeOnly          => CreateCompositeBuffer<TimeOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            BrightDataType.DateOnly          => CreateCompositeBuffer<DateOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
            _                                => throw new ArgumentOutOfRangeException(nameof(type), type, "Unknown table data type")
        };

        /// <summary>
        /// Creates a column analyser
        /// </summary>
        /// <param name="buffer">Buffer to analyse</param>
        /// <param name="metaData"></param>
        /// <param name="maxMetaDataWriteCount">Maximum count to write to meta data</param>
        /// <returns></returns>
        public static IDataAnalyser GetAnalyser(this IReadOnlyBuffer buffer, MetaData metaData, uint maxMetaDataWriteCount = Consts.MaxMetaDataWriteCount)
        {
            var dataType = buffer.DataType.GetBrightDataType();
            var columnType = ColumnTypeClassifier.GetClass(dataType, metaData);
            if (columnType.HasFlag(ColumnClass.Categorical)) {
                if (dataType == BrightDataType.String)
                    return StaticAnalysers.CreateStringAnalyser(maxMetaDataWriteCount);
                return StaticAnalysers.CreateFrequencyAnalyser(buffer.DataType, maxMetaDataWriteCount);
            }

            if (columnType.HasFlag(ColumnClass.IndexBased))
                return StaticAnalysers.CreateIndexAnalyser(maxMetaDataWriteCount);

            if (columnType.HasFlag(ColumnClass.Tensor))
                return StaticAnalysers.CreateDimensionAnalyser();

            return dataType switch
            {
                BrightDataType.Double     => StaticAnalysers.CreateNumericAnalyser(maxMetaDataWriteCount),
                BrightDataType.Float      => StaticAnalysers.CreateNumericAnalyser<float>(maxMetaDataWriteCount),
                BrightDataType.Decimal    => StaticAnalysers.CreateNumericAnalyser<decimal>(maxMetaDataWriteCount),
                BrightDataType.SByte      => StaticAnalysers.CreateNumericAnalyser<sbyte>(maxMetaDataWriteCount),
                BrightDataType.Int        => StaticAnalysers.CreateNumericAnalyser<int>(maxMetaDataWriteCount),
                BrightDataType.Long       => StaticAnalysers.CreateNumericAnalyser<long>(maxMetaDataWriteCount),
                BrightDataType.Short      => StaticAnalysers.CreateNumericAnalyser<short>(maxMetaDataWriteCount),
                BrightDataType.Date       => StaticAnalysers.CreateDateAnalyser(),
                BrightDataType.BinaryData => StaticAnalysers.CreateFrequencyAnalyser<BinaryData>(maxMetaDataWriteCount),
                BrightDataType.DateOnly   => StaticAnalysers.CreateFrequencyAnalyser<DateOnly>(maxMetaDataWriteCount),
                BrightDataType.TimeOnly   => StaticAnalysers.CreateFrequencyAnalyser<TimeOnly>(maxMetaDataWriteCount),
                _                         => throw new NotImplementedException()
            };
        }
        public static IOperation Analyse(this MetaData metaData, bool force, params IReadOnlyBuffer[] buffers)
        {
            if (force || !metaData.Get(Consts.HasBeenAnalysed, false)) {
                if (buffers.Length == 1) {
                    var buffer = buffers[0];
                    var analyser = buffer.GetAnalyser(metaData);
                    return GenericActivator.Create<IOperation>(typeof(BufferScan<>).MakeGenericType(buffer.DataType), buffer, analyser);
                }
                if (buffers.Length > 1) {
                    var analysers = buffers.Select(x => x.GetAnalyser(metaData)).ToArray();
                    var analyser = analysers[0];
                    if (analysers.Skip(1).Any(x => x.GetType() != analyser.GetType()))
                        throw new InvalidOperationException("Expected all buffers to be in same analysis category");

                    var operations = buffers.Select(x => GenericActivator.Create<IOperation>(typeof(BufferScan<>).MakeGenericType(x.DataType), x, analyser)).ToArray();
                    return new AggregateOperation(operations);
                }
            }
            return new NopOperation();
        }
        
        public static async Task<ICompositeBuffer> ToNumeric(this IReadOnlyBuffer buffer, 
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            if(Type.GetTypeCode(buffer.DataType) is TypeCode.DBNull or TypeCode.Empty or TypeCode.Object)
                throw new NotSupportedException();

            var analysis = GenericActivator.Create<ICastToNumericAnalysis>(typeof(CastToNumericAnalysis<>).MakeGenericType(buffer.DataType), buffer);
            await analysis.Process();

            BrightDataType toType;
            if (analysis.IsInteger) {
                toType = analysis switch 
                {
                    { MinValue: >= sbyte.MinValue, MaxValue: <= sbyte.MaxValue } => BrightDataType.SByte,
                    { MinValue: >= short.MinValue, MaxValue: <= short.MaxValue } => BrightDataType.Short,
                    { MinValue: >= int.MinValue, MaxValue: <= int.MaxValue } => BrightDataType.Int,
                    _ => BrightDataType.Long
                };
            } else {
                toType = analysis is { MinValue: >= float.MinValue, MaxValue: <= float.MaxValue } 
                    ? BrightDataType.Float 
                    : BrightDataType.Double;
            }

            var output = GenericActivator.Create<ICompositeBuffer>(typeof(UnmanagedCompositeBuffer<>).MakeGenericType(toType.GetDataType()), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = GenericActivator.Create<IOperation>(typeof(UnmanagedConversion<,>).MakeGenericType(buffer.DataType, toType.GetDataType()), buffer, output);
            await conversion.Process();
            return output;
        }

        static readonly HashSet<string> TrueStrings = new() { "Y", "YES", "TRUE", "T", "1" };
        public static async Task<ICompositeBuffer<bool>> ToBoolean(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
            ) {
            var output = CreateCompositeBuffer<bool>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = (buffer.DataType == typeof(string))
                ? new CustomConversion<string, bool>(StringToBool, (IReadOnlyBuffer<string>)buffer, output)
                : GenericActivator.Create<IOperation>(typeof(UnmanagedConversion<,>).MakeGenericType(buffer.DataType, typeof(bool)), buffer, output)
            ;
            await conversion.Process();
            return output;
            static bool StringToBool(string str) => TrueStrings.Contains(str.ToUpperInvariant());
        }

        public static async Task<ICompositeBuffer<string>> ToString(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = GenericActivator.Create<IOperation>(typeof(ToStringConversion<>).MakeGenericType(buffer.DataType), buffer, output);
            await conversion.Process();
            return output;
        }

        public static async Task<ICompositeBuffer<DateTime>> ToDateTime(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<DateTime>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = (buffer.DataType == typeof(string))
                ? new CustomConversion<string, DateTime>(StringToDate, (IReadOnlyBuffer<string>)buffer, output)
                : GenericActivator.Create<IOperation>(typeof(UnmanagedConversion<,>).MakeGenericType(buffer.DataType, typeof(bool)), buffer, output)
            ;
            await conversion.Process();
            return output;

            static DateTime StringToDate(string str)
            {
                try {
                    return str.ToDateTime();
                }
                catch {
                    // return placeholder date
                    return DateTime.MinValue;
                }
            }
        }

        public static async Task<ICompositeBuffer<DateOnly>> ToDate(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<DateOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = (buffer.DataType == typeof(string))
                ? new CustomConversion<string, DateOnly>(StringToDate, (IReadOnlyBuffer<string>)buffer, output)
                : GenericActivator.Create<IOperation>(typeof(UnmanagedConversion<,>).MakeGenericType(buffer.DataType, typeof(bool)), buffer, output)
            ;
            await conversion.Process();
            return output;

            static DateOnly StringToDate(string str)
            {
                try {
                    return DateOnly.Parse(str);
                }
                catch {
                    // return placeholder date
                    return DateOnly.MinValue;
                }
            }
        }

        public static async Task<ICompositeBuffer<TimeOnly>> ToTime(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<TimeOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = (buffer.DataType == typeof(string))
                ? new CustomConversion<string, TimeOnly>(StringToTime, (IReadOnlyBuffer<string>)buffer, output)
                : GenericActivator.Create<IOperation>(typeof(UnmanagedConversion<,>).MakeGenericType(buffer.DataType, typeof(bool)), buffer, output)
            ;
            await conversion.Process();
            return output;

            static TimeOnly StringToTime(string str)
            {
                try {
                    return TimeOnly.Parse(str);
                }
                catch {
                    // return placeholder date
                    return TimeOnly.MinValue;
                }
            }
        }

        public static async Task<ICompositeBuffer<int>> ToCategoricalIndex(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            
            var output = CreateCompositeBuffer<int>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = GenericActivator.Create<IOperation>(typeof(ToCategoricalIndexConversion<>).MakeGenericType(buffer.DataType), buffer, output);
            await conversion.Process();
            return output;
        }

        public static async Task<ICompositeBuffer<IndexList>> ToIndexList(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<IndexList>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(IndexList))
                conversion = new NopConversion<IndexList>((IReadOnlyBuffer<IndexList>)buffer, output);
            else if (buffer.DataType == typeof(WeightedIndexList))
                conversion = new CustomConversion<WeightedIndexList, IndexList>(WeightedIndexListToIndexList, (IReadOnlyBuffer<WeightedIndexList>)buffer, output);
            else if (buffer.DataType == typeof(ReadOnlyVector))
                conversion = new CustomConversion<ReadOnlyVector, IndexList>(VectorToIndexList, (IReadOnlyBuffer<ReadOnlyVector>)buffer, output);
            else
                throw new NotSupportedException("Only weighted index lists and vectors can be converted to index lists");
            await conversion.Process();
            return output;

            static IndexList VectorToIndexList(ReadOnlyVector vector) => vector.ReadOnlySegment.ToSparse().AsIndexList();
            static IndexList WeightedIndexListToIndexList(WeightedIndexList weightedIndexList) => weightedIndexList.AsIndexList();
        }

        public static async Task<ICompositeBuffer<ReadOnlyVector>> ToVector(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<ReadOnlyVector>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(ReadOnlyVector))
                conversion = new NopConversion<ReadOnlyVector>((IReadOnlyBuffer<ReadOnlyVector>)buffer, output);
            else if (buffer.DataType == typeof(WeightedIndexList))
                conversion = new CustomConversion<WeightedIndexList, ReadOnlyVector>(WeightedIndexListToVector, (IReadOnlyBuffer<WeightedIndexList>)buffer, output);
            else if (buffer.DataType == typeof(IndexList))
                conversion = new CustomConversion<IndexList, ReadOnlyVector>(IndexListToVector, (IReadOnlyBuffer<IndexList>)buffer, output);
            else {
                var index = GenericActivator.Create<IOperation>(typeof(TypedIndexer<>).MakeGenericType(buffer.DataType), buffer);
                await index.Process();
                conversion = GenericActivator.Create<IOperation>(typeof(OneHotConversion<>).MakeGenericType(buffer.DataType), buffer, index, output);
            }
            await conversion.Process();
            return output;

            static ReadOnlyVector WeightedIndexListToVector(WeightedIndexList weightedIndexList) => weightedIndexList.AsDense();
            static ReadOnlyVector IndexListToVector(IndexList indexList) => indexList.AsDense();
        }

        public static async Task<ICompositeBuffer<WeightedIndexList>> ToWeightedIndexList(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<WeightedIndexList>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(WeightedIndexList))
                conversion = new NopConversion<WeightedIndexList>((IReadOnlyBuffer<WeightedIndexList>)buffer, output);
            else if (buffer.DataType == typeof(ReadOnlyVector))
                conversion = new CustomConversion<ReadOnlyVector, WeightedIndexList>(VectorToWeightedIndexList, (IReadOnlyBuffer<ReadOnlyVector>)buffer, output);
            else if (buffer.DataType == typeof(IndexList))
                conversion = new CustomConversion<IndexList, WeightedIndexList>(IndexListToWeightedIndexList, (IReadOnlyBuffer<IndexList>)buffer, output);
            else
                throw new NotSupportedException("Only weighted index lists, index lists and vectors can be converted to vectors");
            await conversion.Process();
            return output;

            static WeightedIndexList IndexListToWeightedIndexList(IndexList indexList) => indexList.AsWeightedIndexList();
            static WeightedIndexList VectorToWeightedIndexList(ReadOnlyVector vector) => vector.ToSparse();
        }

        public static async Task<ICompositeBuffer<T>> To<T>(this IReadOnlyBuffer buffer,
            IProvideTempData? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null) where T: unmanaged
        {
            var output = CreateCompositeBuffer<T>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = GenericActivator.Create<IOperation>(typeof(UnmanagedConversion<,>).MakeGenericType(buffer.DataType, typeof(T)), buffer, output);
            await conversion.Process();
            return output;
        }

        public static async Task<ICompositeBuffer<ReadOnlyVector>> Vectorise(this IReadOnlyBuffer[] buffers,
            IProvideTempData? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<ReadOnlyVector>(x => new(x), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var floatBuffers = buffers.Select(x => x.ConvertTo<float>());
            var conversion = new ManyToOneMutation<float, ReadOnlyVector>(floatBuffers, output, x => new(x));
            await conversion.Process();
            return output;
        }

        public static async Task<T[]> ToArray<T>(this IReadOnlyBuffer buffer) where T : notnull
        {
            var ret = new T[buffer.Size];
            var index = 0;
            await foreach(var item in buffer.GetValues<T>())
                ret[index++] = item;
            return ret;
        }
    }
}
