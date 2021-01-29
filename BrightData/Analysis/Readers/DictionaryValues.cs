﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BrightData.Analysis.Readers
{
    /// <summary>
    /// Dictionary that holds category string indices
    /// </summary>
    public class DictionaryValues
    {
        readonly List<string> _table;

        internal DictionaryValues(IMetaData metaData)
        {
            var len = Consts.CategoryPrefix.Length;
            _table = metaData.GetStringsWithPrefix(Consts.CategoryPrefix)
                .Select(key => (Key: Int32.Parse(key[len..]), Value: metaData.Get(key, "")))
                .OrderBy(d => d.Key)
                .Select(d => d.Value)
                .ToList();
        }

        /// <summary>
        /// Converts from category indices to string
        /// </summary>
        /// <param name="categoryIndices"></param>
        /// <returns></returns>
        public IEnumerable<string> GetValues(IEnumerable<int> categoryIndices) => categoryIndices.Select(i => _table[i]);

        /// <summary>
        /// Gets the string associated with a category index
        /// </summary>
        /// <param name="categoryIndex"></param>
        /// <returns></returns>
        public string GetValue(int categoryIndex) => _table[categoryIndex];
    }
}
