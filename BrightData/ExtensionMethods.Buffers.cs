﻿using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using BrightData.Buffer.Composite;
using BrightData.Buffer.Operations;
using BrightData.Buffer.Operations.Conversion;
using BrightData.Buffer.ReadOnly;
using BrightData.Buffer.ReadOnly.Converter;
using BrightData.Converter;
using BrightData.Helper;
using BrightData.LinearAlgebra.ReadOnly;
using CommunityToolkit.HighPerformance.Buffers;
using BrightData.Types;

namespace BrightData
{
    public partial class ExtensionMethods
    {
        /// <summary>
        /// Enumerates values in the buffer (blocking)
        /// </summary>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public static IEnumerable<object> GetValues(this IReadOnlyBuffer buffer)
        {
            return buffer.EnumerateAll().ToBlockingEnumerable();
        }

        /// <summary>
        /// Async enumeration of values in the buffer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentException"></exception>
        public static IAsyncEnumerable<T> GetValues<T>(this IReadOnlyBuffer buffer) where T: notnull
        {
            if (buffer.DataType != typeof(T))
                throw new ArgumentException($"Buffer is of type {buffer.DataType} but requested {typeof(T)}");
            var typedBuffer = (IReadOnlyBuffer<T>)buffer;
            return typedBuffer.EnumerateAllTyped();
        }

        /// <summary>
        /// Casts or converts the buffer to a string buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer<string> ToReadOnlyStringBuffer(this IReadOnlyBuffer buffer)
        {
            if (buffer.DataType == typeof(string))
                return (IReadOnlyBuffer<string>)buffer;
            return GenericTypeMapping.ToStringConverter(buffer);

        }

        /// <summary>
        /// Creates a new buffer in which each value is converted via the conversion function
        /// </summary>
        /// <typeparam name="FT"></typeparam>
        /// <typeparam name="TT"></typeparam>
        /// <param name="buffer"></param>
        /// <param name="converter"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer<TT> Convert<FT, TT>(this IReadOnlyBuffer<FT> buffer, Func<FT, TT> converter)
            where FT: notnull
            where TT: notnull
        {
            return (IReadOnlyBuffer<TT>)GenericTypeMapping.TypeConverter(typeof(TT), buffer, new CustomConversionFunction<FT, TT>(converter));
        }

        /// <summary>
        /// Finds distinct groups within the buffers based on string comparison of the concatenated values
        /// </summary>
        /// <param name="buffers"></param>
        /// <returns></returns>
        public static async Task<Dictionary<string /* group */, List<uint> /* row indices in group */>> GetGroups(this IReadOnlyBuffer[] buffers)
        {
            // ReSharper disable once NotDisposedResourceIsReturned
            var enumerators = buffers.Select(x => x.EnumerateAll().GetAsyncEnumerator()).ToArray();
            var shouldContinue = true;
            var sb = new StringBuilder();
            var ret = new Dictionary<string, List<uint>>();
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

                if (shouldContinue) {
                    var str = sb.ToString();
                    if (!ret.TryGetValue(str, out var list))
                        ret.Add(str, list = []);
                    list.Add(rowIndex++);
                }
            }

            foreach (var enumerator in enumerators)
                await enumerator.DisposeAsync();

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

        /// <summary>
        /// Converts the buffer to a read only sequence
        /// </summary>
        /// <param name="buffer"></param>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sequence"></param>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public static SequenceReader<T> AsSequenceReader<T>(this ReadOnlySequence<T> sequence) where T : unmanaged, IEquatable<T>
        {
            return new SequenceReader<T>(sequence);
        }

        /// <summary>
        /// Retrieves an item from the buffer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public static async Task<T> GetItem<T>(this IReadOnlyBuffer<T> buffer, uint index) where T: notnull
        {
            var blockIndex = index / buffer.BlockSize;
            var blockMemory = await buffer.GetTypedBlock(blockIndex);
            var ret = blockMemory.Span[(int)(index % buffer.BlockSize)];
            return ret;
        }

        /// <summary>
        /// Retrieves items from the buffer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public static async Task<T[]> GetItems<T>(this IReadOnlyBuffer<T> buffer, uint[] indices) where T: notnull
        {
            var blocks = indices.Select((x, i) => (TargetIndex: (uint)i, BlockIndex: x / buffer.BlockSize, RelativeIndex: x % buffer.BlockSize))
                    .GroupBy(x => x.BlockIndex)
                    .OrderBy(x => x.Key)
                ;
            var ret = new T[indices.Length];
            foreach (var block in blocks) {
                var blockMemory = await buffer.GetTypedBlock(block.Key);
                AddIndexedItems(blockMemory, block, ret);
            }
            return ret;

            static void AddIndexedItems(ReadOnlyMemory<T> data, IEnumerable<(uint Index, uint BlockIndex, uint RelativeIndex)> list, T[] output)
            {
                var span = data.Span;
                foreach (var (targetIndex, _, relativeIndex) in list)
                    output[targetIndex] = span[(int)relativeIndex];
            }
        }

        ///// <summary>
        ///// Buffer iterator
        ///// </summary>
        ///// <typeparam name="T"></typeparam>
        //public ref struct ReadOnlyBufferIterator<T> where T: notnull
        //{
        //    readonly IReadOnlyBuffer<T> _buffer;
        //    ReadOnlyMemory<T> _currentBlock = ReadOnlyMemory<T>.Empty;
        //    uint _blockIndex = 0, _position = 0;

        //    internal ReadOnlyBufferIterator(IReadOnlyBuffer<T> buffer) => _buffer = buffer;

        //    /// <summary>
        //    /// Advances to the next position
        //    /// </summary>
        //    /// <returns></returns>
        //    public bool MoveNext()
        //    {
        //        if (++_position < _currentBlock.Length)
        //            return true;

        //        while(_blockIndex < _buffer.BlockCount) {
        //            _currentBlock = _buffer.GetTypedBlock(_blockIndex++).Result;
        //            if (_currentBlock.Length > 0) {
        //                _position = 0;
        //                return true;
        //            }
        //        }
        //        return false;
        //    }

        //    /// <summary>
        //    /// Current iterator value
        //    /// </summary>
        //    public readonly ref readonly T Current => ref _currentBlock.Span[(int)_position];

        //    /// <summary>
        //    /// Converts to enumerator
        //    /// </summary>
        //    /// <returns></returns>
        //    public readonly ReadOnlyBufferIterator<T> GetEnumerator() => this;
        //}

        ///// <summary>
        ///// Creates an iterator for the buffer
        ///// </summary>
        ///// <param name="buffer"></param>
        ///// <typeparam name="T"></typeparam>
        ///// <returns></returns>
        //public static ReadOnlyBufferIterator<T> GetEnumerator<T>(this IReadOnlyBuffer<T> buffer) where T: notnull => new(buffer);

        /// <summary>
        /// Converts the buffer to an array
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <returns></returns>
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

        /// <summary>
        /// Creates a composite buffer for strings
        /// </summary>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static ICompositeBuffer<string> CreateCompositeBuffer(
            this IProvideDataBlocks? tempStreams, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) => new StringCompositeBuffer(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        /// <summary>
        /// Creates a composite buffer for types that can be created from a block of byte data
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tempStreams"></param>
        /// <param name="createItem"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static ICompositeBuffer<T> CreateCompositeBuffer<T>(
            this IProvideDataBlocks? tempStreams,
            CreateFromReadOnlyByteSpan<T> createItem,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null) where T: IHaveDataAsReadOnlyByteSpan => new ManagedCompositeBuffer<T>(createItem, tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        /// <summary>
        /// Creates a composite buffer for unmanaged types
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static ICompositeBuffer<T> CreateCompositeBuffer<T>(
            this IProvideDataBlocks? tempStreams, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) where T: unmanaged => new UnmanagedCompositeBuffer<T>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        /// <summary>
        /// Creates a composite buffer
        /// </summary>
        /// <param name="dataType"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public static ICompositeBuffer CreateCompositeBuffer(
            this BrightDataType dataType,
            IProvideDataBlocks? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null)
        {
            return dataType switch {
                BrightDataType.BinaryData        => CreateCompositeBuffer<BinaryData>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Boolean           => CreateCompositeBuffer<bool>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Date              => CreateCompositeBuffer<DateTime>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.DateOnly          => CreateCompositeBuffer<DateOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Decimal           => CreateCompositeBuffer<decimal>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.SByte             => CreateCompositeBuffer<sbyte>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Short             => CreateCompositeBuffer<short>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Int               => CreateCompositeBuffer<int>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Long              => CreateCompositeBuffer<long>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Float             => CreateCompositeBuffer<float>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Double            => CreateCompositeBuffer<double>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.String            => CreateCompositeBuffer(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.IndexList         => CreateCompositeBuffer<IndexList>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.WeightedIndexList => CreateCompositeBuffer<WeightedIndexList>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Vector            => CreateCompositeBuffer<ReadOnlyVector>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Matrix            => CreateCompositeBuffer<ReadOnlyMatrix>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Tensor3D          => CreateCompositeBuffer<ReadOnlyTensor3D>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.Tensor4D          => CreateCompositeBuffer<ReadOnlyTensor4D>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems),
                BrightDataType.TimeOnly          => CreateCompositeBuffer<TimeOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems),
                _                                => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, $"Not able to create a composite buffer for type: {dataType}")
            };
        }

        /// <summary>
        /// Creates a buffer writer from a composite buffer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <param name="bufferSize"></param>
        /// <returns></returns>
        public static IBufferWriter<T> AsBufferWriter<T>(this ICompositeBuffer<T> buffer, int bufferSize = 256) where T : notnull => new CompositeBufferWriter<T>(buffer, bufferSize);

        /// <summary>
        /// Returns true of the buffer can be encoded (distinct items mapped to indices)
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <returns></returns>
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
            IProvideDataBlocks? tempStreams = null, 
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
                    data.Append(index);
                }
                else if (len > 1) {
                    var spanOwner = SpanOwner<uint>.Empty;
                    var indices = len <= Consts.MaxStackAllocSizeInBytes / sizeof(uint)
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
                        data.Append(indices);
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

        /// <summary>
        /// Creates a composite buffer for the type
        /// </summary>
        /// <param name="type"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static ICompositeBuffer CreateCompositeBuffer(this Type type,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) => CreateCompositeBuffer(GetBrightDataType(type), tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

        /// <summary>
        /// Creates a column analyser
        /// </summary>
        /// <param name="buffer">Buffer to analyse</param>
        /// <param name="metaData"></param>
        /// <param name="maxMetaDataWriteCount">Maximum count to write to meta data</param>
        /// <returns></returns>
        public static IDataAnalyser GetAnalyser(this IReadOnlyBuffer buffer, MetaData metaData, uint maxMetaDataWriteCount = Consts.MaxMetaDataWriteCount)
        {
            return buffer.DataType.GetBrightDataType().GetAnalyser(metaData, maxMetaDataWriteCount);
        }

        /// <summary>
        /// Analyses the buffer
        /// </summary>
        /// <param name="metaData"></param>
        /// <param name="force"></param>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public static IOperation Analyse(this MetaData metaData, bool force, IReadOnlyBuffer buffer)
        {
            if (force || !metaData.Get(Consts.HasBeenAnalysed, false)) {
                var analyser = buffer.GetAnalyser(metaData);
                void WriteToMetaData() => analyser.WriteTo(metaData);
                return buffer.CreateBufferCopyOperation(analyser, WriteToMetaData);
            }
            return new NopOperation();
        }

        /// <summary>
        /// Creates an operation that copies the blocks in the buffer to a destination 
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="output"></param>
        /// <param name="action"></param>
        /// <returns></returns>
        public static IOperation CreateBufferCopyOperation(this IReadOnlyBuffer buffer, IAppendBlocks output, Action? action = null)
        {
            return buffer.DataType.GetBrightDataType() switch {
                BrightDataType.IndexList         => CastBuffer<IndexList, IHaveIndices>(buffer, output, action),
                BrightDataType.WeightedIndexList => CastBuffer<WeightedIndexList, IHaveIndices>(buffer, output, action),
                BrightDataType.Vector            => CastBuffer<ReadOnlyVector, IReadOnlyTensor>(buffer, output, action),
                BrightDataType.Matrix            => CastBuffer<ReadOnlyMatrix, IReadOnlyTensor>(buffer, output, action),
                BrightDataType.Tensor3D          => CastBuffer<ReadOnlyTensor3D, IReadOnlyTensor>(buffer, output, action),
                BrightDataType.Tensor4D          => CastBuffer<ReadOnlyTensor4D, IReadOnlyTensor>(buffer, output, action),
                _                                => GenericTypeMapping.BufferCopyOperation(buffer, output, action)
            };

            static BufferCopyOperation<CT2> CastBuffer<T2, CT2>(IReadOnlyBuffer buffer, IAppendBlocks analyser, Action? action = null) where T2 : notnull where CT2 : notnull
            {
                var buffer2 = (IReadOnlyBuffer<T2>)buffer;
                var buffer3 = buffer2.Cast<T2, CT2>();
                var dataAnalyser2 = (IAppendBlocks<CT2>)analyser;
                return new BufferCopyOperation<CT2>(buffer3, dataAnalyser2, action);
            }
        }

        /// <summary>
        /// Analyse the buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="force"></param>
        /// <returns></returns>
        public static IOperation Analyse(this IReadOnlyBufferWithMetaData buffer, bool force) => Analyse(buffer.MetaData, force, buffer);
        
        /// <summary>
        /// Creates a numeric composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException"></exception>
        public static async Task<ICompositeBuffer> ToNumeric(this IReadOnlyBuffer buffer, 
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            if(Type.GetTypeCode(buffer.DataType) is TypeCode.DBNull or TypeCode.Empty or TypeCode.Object)
                throw new NotSupportedException();

            // convert from strings
            if (buffer.DataType == typeof(string))
                buffer = buffer.ConvertTo<double>();

            var analysis = GenericTypeMapping.SimpleNumericAnalysis(buffer);
            await analysis.Execute();

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

            var output = CreateCompositeBuffer(toType, tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var converter = buffer.ConvertUnmanagedTo(toType.GetDataType());
            var conversion = GenericTypeMapping.BufferCopyOperation(converter, output);
            await conversion.Execute();
            return output;
        }

        static readonly HashSet<string> TrueStrings = ["Y", "YES", "TRUE", "T", "1"];

        /// <summary>
        /// Creates a boolean composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<bool>> ToBoolean(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
            ) {
            var output = CreateCompositeBuffer<bool>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(bool))
                conversion = buffer.CreateBufferCopyOperation(output);
            else if (buffer.DataType == typeof(string))
                conversion = new CustomConversion<string, bool>(StringToBool, buffer.ToReadOnlyStringBuffer(), output);
            else {
                var converted = buffer.ConvertUnmanagedTo(typeof(bool));
                conversion = GenericTypeMapping.BufferCopyOperation(converted, output);
            }

            await conversion.Execute();
            return output;
            static bool StringToBool(string str) => TrueStrings.Contains(str.ToUpperInvariant());
        }

        /// <summary>
        /// Creates a string composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<string>> ToString(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var conversion = buffer.DataType == typeof(string)
                ? buffer.CreateBufferCopyOperation(output) 
                : new BufferCopyOperation<string>(GenericTypeMapping.ToStringConverter(buffer), output, null);
            await conversion.Execute();
            return output;
        }

        /// <summary>
        /// Creates a date time composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<DateTime>> ToDateTime(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<DateTime>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(DateTime))
                conversion = buffer.CreateBufferCopyOperation(output);
            else if (buffer.DataType == typeof(string))
                conversion = new CustomConversion<string, DateTime>(StringToDate, buffer.ToReadOnlyStringBuffer(), output);
            else {
                var converted = buffer.ConvertUnmanagedTo(typeof(DateTime));
                conversion = GenericTypeMapping.BufferCopyOperation(converted, output);
            }

            await conversion.Execute();
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

        /// <summary>
        /// Creates a date composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<DateOnly>> ToDate(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<DateOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(DateOnly))
                conversion = buffer.CreateBufferCopyOperation(output);
            else if (buffer.DataType == typeof(string))
                conversion = new CustomConversion<string, DateOnly>(StringToDate, buffer.ToReadOnlyStringBuffer(), output);
            else {
                var converted = buffer.ConvertUnmanagedTo(typeof(DateOnly));
                conversion = GenericTypeMapping.BufferCopyOperation(converted, output);
            }

            await conversion.Execute();
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

        /// <summary>
        /// Creates a time composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<TimeOnly>> ToTime(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<TimeOnly>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(TimeOnly))
                conversion = buffer.CreateBufferCopyOperation(output);
            else if (buffer.DataType == typeof(string))
                conversion = new CustomConversion<string, TimeOnly>(StringToTime, buffer.ToReadOnlyStringBuffer(), output);
            else {
                var converted = buffer.ConvertUnmanagedTo(typeof(TimeOnly));
                conversion = GenericTypeMapping.BufferCopyOperation(converted, output);
            }

            await conversion.Execute();
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

        /// <summary>
        /// Creates a categorical index composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<int>> ToCategoricalIndex(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            
            var output = CreateCompositeBuffer<int>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);
            var index = GenericTypeMapping.TypedIndexer(buffer);
            await index.Execute();
            var indexer = (ICanIndex)index;
            var mapping = indexer.GetMapping();

            var metaData = output.MetaData;
            metaData.SetIsCategorical(true);
            foreach (var category in mapping.OrderBy(d => d.Value))
                metaData.Set(Consts.CategoryPrefix + category.Value, category.Key);

            var categories = GenericTypeMapping.CategoricalIndexConverter(buffer, indexer);
            var conversion = GenericTypeMapping.BufferCopyOperation(categories, output);
            await conversion.Execute();
            return output;
        }

        /// <summary>
        /// Creates an index list composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException"></exception>
        public static async Task<ICompositeBuffer<IndexList>> ToIndexList(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<IndexList>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(IndexList))
                conversion = new NopConversion<IndexList>((IReadOnlyBuffer<IndexList>)buffer, output);
            else if (buffer.DataType == typeof(WeightedIndexList))
                conversion = new CustomConversion<WeightedIndexList, IndexList>(WeightedIndexListToIndexList, (IReadOnlyBuffer<WeightedIndexList>)buffer, output);
            else if (buffer.DataType == typeof(ReadOnlyVector))
                conversion = new CustomConversion<ReadOnlyVector, IndexList>(VectorToIndexList, (IReadOnlyBuffer<ReadOnlyVector>)buffer, output);
            else
                throw new NotSupportedException("Only weighted index lists and vectors can be converted to index lists");
            await conversion.Execute();
            return output;

            static IndexList VectorToIndexList(ReadOnlyVector vector) => vector.ReadOnlySegment.ToSparse().AsIndexList();
            static IndexList WeightedIndexListToIndexList(WeightedIndexList weightedIndexList) => weightedIndexList.AsIndexList();
        }

        /// <summary>
        /// Creates a vector composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<ReadOnlyVector>> ToVector(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<ReadOnlyVector>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(ReadOnlyVector))
                conversion = new NopConversion<ReadOnlyVector>((IReadOnlyBuffer<ReadOnlyVector>)buffer, output);
            else if (buffer.DataType == typeof(WeightedIndexList))
                conversion = new CustomConversion<WeightedIndexList, ReadOnlyVector>(WeightedIndexListToVector, (IReadOnlyBuffer<WeightedIndexList>)buffer, output);
            else if (buffer.DataType == typeof(IndexList))
                conversion = new CustomConversion<IndexList, ReadOnlyVector>(IndexListToVector, (IReadOnlyBuffer<IndexList>)buffer, output);
            else {
                var index = GenericTypeMapping.TypedIndexer(buffer);
                await index.Execute();
                var indexer = (ICanIndex)index;
                var vectorBuffer = GenericTypeMapping.OneHotConverter(buffer, indexer);
                conversion = GenericTypeMapping.BufferCopyOperation(vectorBuffer, output);
                var categoryIndex = indexer.GetMapping();

                var metaData = output.MetaData;
                metaData.SetIsOneHot(true);
                foreach (var category in categoryIndex)
                    metaData.Set(Consts.CategoryPrefix + category.Value, category.Key);
            }
            await conversion.Execute();
            return output;

            static ReadOnlyVector WeightedIndexListToVector(WeightedIndexList weightedIndexList) => weightedIndexList.AsDense();
            static ReadOnlyVector IndexListToVector(IndexList indexList) => indexList.AsDense();
        }

        /// <summary>
        /// Creates a weighted index list composite buffer from an existing buffer
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException"></exception>
        public static async Task<ICompositeBuffer<WeightedIndexList>> ToWeightedIndexList(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null, 
            int blockSize = Consts.DefaultBlockSize, 
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<WeightedIndexList>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems);
            IOperation conversion;
            if (buffer.DataType == typeof(WeightedIndexList))
                conversion = new NopConversion<WeightedIndexList>((IReadOnlyBuffer<WeightedIndexList>)buffer, output);
            else if (buffer.DataType == typeof(ReadOnlyVector))
                conversion = new CustomConversion<ReadOnlyVector, WeightedIndexList>(VectorToWeightedIndexList, (IReadOnlyBuffer<ReadOnlyVector>)buffer, output);
            else if (buffer.DataType == typeof(IndexList))
                conversion = new CustomConversion<IndexList, WeightedIndexList>(IndexListToWeightedIndexList, (IReadOnlyBuffer<IndexList>)buffer, output);
            else
                throw new NotSupportedException("Only weighted index lists, index lists and vectors can be converted to vectors");
            await conversion.Execute();
            return output;

            static WeightedIndexList IndexListToWeightedIndexList(IndexList indexList) => indexList.AsWeightedIndexList();
            static WeightedIndexList VectorToWeightedIndexList(ReadOnlyVector vector) => vector.ToSparse();
        }

        /// <summary>
        /// Creates a typed composite buffer from an existing buffer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<T>> To<T>(this IReadOnlyBuffer buffer,
            IProvideDataBlocks? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null) where T: unmanaged
        {
            var output = CreateCompositeBuffer<T>(tempStreams, blockSize, maxInMemoryBlocks, maxDistinctItems);

            // convert from strings
            if (buffer.DataType == typeof(string))
                buffer = buffer.ConvertTo<double>();

            var converted = buffer.ConvertUnmanagedTo(typeof(T));
            var conversion = GenericTypeMapping.BufferCopyOperation(converted, output);
            await conversion.Execute();
            return output;
        }

        /// <summary>
        /// Vectorise the buffers
        /// </summary>
        /// <param name="buffers"></param>
        /// <param name="tempStreams"></param>
        /// <param name="blockSize"></param>
        /// <param name="maxInMemoryBlocks"></param>
        /// <param name="maxDistinctItems"></param>
        /// <returns></returns>
        public static async Task<ICompositeBuffer<ReadOnlyVector>> Vectorise(this IReadOnlyBuffer[] buffers,
            IProvideDataBlocks? tempStreams = null,
            int blockSize = Consts.DefaultBlockSize,
            uint? maxInMemoryBlocks = null,
            uint? maxDistinctItems = null
        ) {
            var output = CreateCompositeBuffer<ReadOnlyVector>(tempStreams, x => new(x), blockSize, maxInMemoryBlocks, maxDistinctItems);
            var floatBuffers = buffers.Select(x => x.ConvertTo<float>());
            var conversion = new ManyToOneMutation<float, ReadOnlyVector>(floatBuffers, output, x => new(x));
            await conversion.Execute();
            return output;
        }

        /// <summary>
        /// Creates an array from the buffer
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public static async Task<T[]> ToArray<T>(this IReadOnlyBuffer buffer) where T : notnull
        {
            var ret = new T[buffer.Size];
            var index = 0;
            await foreach(var item in buffer.GetValues<T>())
                ret[index++] = item;
            return ret;
        }

        /// <summary>
        /// Creates a read only string composite buffer from a stream
        /// </summary>
        /// <param name="stream"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer<string> GetReadOnlyStringCompositeBuffer(this Stream stream)
        {
            return new ReadOnlyStringCompositeBuffer(stream);
        }

        /// <summary>
        /// Creates a read only composite buffer for unmanaged types from a stream
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="stream"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer<T> GetReadOnlyCompositeBuffer<T>(this Stream stream) where T: unmanaged
        {
            return new ReadOnlyUnmanagedCompositeBuffer<T>(stream);
        }

        /// <summary>
        /// Creates a read only composite buffer for types that can be initialised from a byte block from a stream
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="stream"></param>
        /// <param name="createItem"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer<T> GetReadOnlyCompositeBuffer<T>(this Stream stream, CreateFromReadOnlyByteSpan<T> createItem) where T : IHaveDataAsReadOnlyByteSpan
        {
            return new ReadOnlyManagedCompositeBuffer<T>(createItem, stream);
        }

        /// <summary>
        /// Casts a buffer to another type
        /// </summary>
        /// <typeparam name="FT"></typeparam>
        /// <typeparam name="TT"></typeparam>
        /// <param name="buffer"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer<TT> Cast<FT, TT>(this IReadOnlyBuffer<FT> buffer) where FT : notnull where TT : notnull
        {
            return new CastConverter<FT, TT>(buffer);
        }

        /// <summary>
        /// Converts the buffer that contains an unmanaged type to the specified type
        /// </summary>
        /// <param name="buffer"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public static IReadOnlyBuffer ConvertUnmanagedTo(this IReadOnlyBuffer buffer, Type type)
        {
            var converter = StaticConverters.GetConverter(buffer.DataType, type);
            return GenericTypeMapping.TypeConverter(type, buffer, converter);
        }
    }
}
