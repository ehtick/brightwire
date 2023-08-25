﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BrightData.Table.Buffer.ReadOnly
{
    public delegate ReadOnlyMemory<T> BlockMapper<FT, T>(ReadOnlySpan<FT> span);
    internal class MappedReadOnlyBuffer<IT, T> : IReadOnlyBufferWithMetaData<T> 
        where IT : notnull
        where T : notnull 
    {
        readonly IReadOnlyBufferWithMetaData<IT> _index;
        readonly BlockMapper<IT, T> _mapper;
        ReadOnlyMemory<T>? _lastBlock;
        uint _lastBlockIndex;

        public MappedReadOnlyBuffer(IReadOnlyBufferWithMetaData<IT> index, BlockMapper<IT, T> mapper)
        {
            _index = index;
            _mapper = mapper;
        }

        public uint BlockSize => _index.BlockSize;
        public uint BlockCount => _index.BlockCount;
        public Type DataType => typeof(T);

        public async Task ForEachBlock(BlockCallback<T> callback)
        {
            await _index.ForEachBlock(x => {
                var mapped = _mapper(x);
                callback(mapped.Span);
            });
        }

        public async Task<ReadOnlyMemory<T>> GetBlock(uint blockIndex)
        {
            if (blockIndex >= BlockCount)
                return ReadOnlyMemory<T>.Empty;
            if (_lastBlockIndex == blockIndex && _lastBlock.HasValue)
                return _lastBlock.Value;

            _lastBlockIndex = blockIndex;
            var indices = await _index.GetBlock(blockIndex);
            var ret = _mapper(indices.Span);
            _lastBlock = ret;
            return ret;
        }

        public IAsyncEnumerator<T> GetAsyncEnumerator() => EnumerateAll().GetAsyncEnumerator();

        public async IAsyncEnumerable<T> EnumerateAll()
        {
            for (uint i = 0; i < BlockCount; i++) {
                var block = await GetBlock(i);
                for (var j = 0; j < block.Length; j++)
                    yield return block.Span[j];
            }
        }

        public MetaData MetaData => _index.MetaData;
    }
}
