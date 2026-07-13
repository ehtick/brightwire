using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using BrightData;
using BrightData.DataTable.Columns;
using BrightData.DataTable.Rows;
using BrightData.Helper;
using BrightWire.Models.TreeBased;

namespace BrightWire.TreeBased.Training
{
    /// <summary>
    /// Trains decision tree models using an entropy-based splitting criterion.
    /// Implements the ID3/CART family of algorithms where the best split at each node
    /// is chosen by maximizing information gain (reduction in entropy).
    /// </summary>
    /// <remarks>
    /// Reference: https://en.wikipedia.org/wiki/Decision_tree_learning
    /// </remarks>
    public static class DecisionTreeTrainer
    {
        /// <summary>
        /// Represents a single attribute split during tree construction.
        /// A split is either a categorical value match or a numeric threshold comparison.
        /// </summary>
        private sealed class Attribute
        {
            private readonly uint _columnIndex;
            private readonly string? _category;
            private readonly double? _split;

            /// <summary>
            /// Creates a categorical attribute split that matches a specific category value.
            /// </summary>
            /// <param name="columnIndex">The column to split on.</param>
            /// <param name="category">The category value to match.</param>
            public Attribute(uint columnIndex, string category)
            {
                _columnIndex = columnIndex;
                _category = category;
                _split = null;
            }

            /// <summary>
            /// Creates a numeric attribute split using a threshold comparison.
            /// </summary>
            /// <param name="columnIndex">The column to split on.</param>
            /// <param name="split">The threshold value. Values below go left, values above go right.</param>
            public Attribute(uint columnIndex, double split)
            {
                _columnIndex = columnIndex;
                _category = null;
                _split = split;
            }

            /// <summary>
            /// Gets the column index this attribute split operates on.
            /// </summary>
            public uint ColumnIndex => _columnIndex;

            /// <summary>
            /// Gets the category value for categorical splits, or null for numeric splits.
            /// </summary>
            public string? Category => _category;

            /// <summary>
            /// Gets the threshold value for numeric splits, or null for categorical splits.
            /// </summary>
            public double? Split => _split;

            /// <inheritdoc />
            public override bool Equals(object? obj)
            {
                if (obj is Attribute other)
                    return other._columnIndex == _columnIndex && other._category == _category && Math<double>.AreApproximatelyEqual(_split, other._split);
                return false;
            }

            /// <inheritdoc />
            public override int GetHashCode()
            {
                if (_category != null)
                    return _columnIndex.GetHashCode() ^ _category.GetHashCode();
                return _columnIndex.GetHashCode() ^ _split!.GetHashCode();
            }

            /// <inheritdoc />
            public override string ToString()
            {
                if (_category != null)
                    return $"{_category} ({_columnIndex})";
                return $"threshold: {_split} ({_columnIndex})";
            }

            /// <summary>
            /// Partitions the given rows into groups based on this attribute split.
            /// </summary>
            /// <param name="rows">The rows to partition.</param>
            /// <returns>A dictionary mapping partition labels to the rows belonging to each partition.</returns>
            public IReadOnlyDictionary<string, List<InMemoryRow>> Partition(IEnumerable<InMemoryRow> rows)
            {
                List<InMemoryRow>? temp;
                var ret = new Dictionary<string, List<InMemoryRow>>();
                if (_category != null)
                {
                    foreach (var item in rows)
                    {
                        var val = item.GetCategory(_columnIndex);
                        if (!ret.TryGetValue(val, out temp))
                            ret.Add(val, temp = []);
                        temp.Add(item);
                    }
                }
                else
                {
                    var splitVal = _split ?? 0;
                    foreach (var item in rows)
                    {
                        var val = item.GetValue(_columnIndex);
                        var label = val < splitVal ? LabelBelow : LabelAbove;
                        if (!ret.TryGetValue(label, out temp))
                            ret.Add(label, temp = []);
                        temp.Add(item);
                    }
                }
                return ret;
            }

            /// <summary>
            /// Label used for rows below the numeric threshold.
            /// </summary>
            private const string LabelBelow = "below";

            /// <summary>
            /// Label used for rows at or above the numeric threshold.
            /// </summary>
            private const string LabelAbove = "above";
        }

        /// <summary>
        /// An in-memory representation of a single training row.
        /// Stores pre-extracted categorical and continuous values for fast access during tree construction.
        /// </summary>
        private sealed class InMemoryRow
        {
            private readonly Dictionary<uint, string> _category = [];
            private readonly Dictionary<uint, double> _continuous = [];

            /// <summary>
            /// Creates an in-memory row by extracting the relevant columns from a data table row.
            /// </summary>
            /// <param name="row">The source data table row.</param>
            /// <param name="categorical">Column indices that are categorical.</param>
            /// <param name="continuous">Column indices that are continuous/numeric.</param>
            /// <param name="classColumnIndex">The index of the classification label column.</param>
            public InMemoryRow(GenericTableRow row, HashSet<uint> categorical, HashSet<uint> continuous, uint classColumnIndex)
            {
                ClassificationLabel = row.Get<string>(classColumnIndex);
                foreach (var columnIndex in categorical)
                    _category.Add(columnIndex, row.Get<string>(columnIndex));
                foreach (var columnIndex in continuous)
                    _continuous.Add(columnIndex, row.Get<double>(columnIndex));
            }

            /// <summary>
            /// Gets the classification label for this row.
            /// </summary>
            public string ClassificationLabel { get; }

            /// <summary>
            /// Gets the categorical value for the specified column.
            /// </summary>
            /// <param name="columnIndex">The column index.</param>
            /// <returns>The category string.</returns>
            public string GetCategory(uint columnIndex)
            {
                return _category[columnIndex];
            }

            /// <summary>
            /// Gets the numeric value for the specified column.
            /// </summary>
            /// <param name="columnIndex">The column index.</param>
            /// <returns>The numeric value.</returns>
            public double GetValue(uint columnIndex)
            {
                return _continuous[columnIndex];
            }
        }

        /// <summary>
        /// Holds metadata about the training data table, including column classifications and in-memory row data.
        /// </summary>
        private sealed class TableInfo
        {
            private readonly HashSet<uint> _categorical = [];
            private readonly HashSet<uint> _continuous = [];

            /// <summary>
            /// Analyzes the data table and extracts all rows into memory for fast access during tree construction.
            /// </summary>
            /// <param name="table">The training data table.</param>
            public TableInfo(IDataTable table)
            {
                ClassColumnIndex = table.GetTargetColumnOrThrow();
                var metaData = table.ColumnMetaData;
                for (uint i = 0, len = table.ColumnCount; i < len; i++)
                {
                    if (i != ClassColumnIndex)
                    {
                        var columnType = table.ColumnTypes[i];
                        var columnMetaData = metaData[i];
                        var columnClass = ColumnTypeClassifier.GetClass(columnType, columnMetaData);
                        if ((columnClass & ColumnClass.Categorical) != 0)
                            _categorical.Add(i);
                        else if ((columnClass & ColumnClass.Numeric) != 0)
                            _continuous.Add(i);
                    }
                }
                foreach (var row in table.EnumerateRows().ToBlockingEnumerable())
                {
                    Data.Add(new InMemoryRow(row, _categorical, _continuous, ClassColumnIndex));
                }
            }

            /// <summary>
            /// Gets the collection of categorical column indices.
            /// </summary>
            public IEnumerable<uint> CategoricalColumns => _categorical;

            /// <summary>
            /// Gets the collection of continuous (numeric) column indices.
            /// </summary>
            public IEnumerable<uint> ContinuousColumns => _continuous;

            /// <summary>
            /// Gets the in-memory representation of all training rows.
            /// </summary>
            public List<InMemoryRow> Data { get; } = [];

            /// <summary>
            /// Gets the column index of the classification label column.
            /// </summary>
            public uint ClassColumnIndex { get; }
        }

        /// <summary>
        /// Represents a single node in the decision tree during construction.
        /// Tracks the data subset, computed attributes, entropy, and parent-child relationships.
        /// </summary>
        private class Node
        {
            private readonly Dictionary<string, int> _classCount;
            private Node? _parent;
            private Attribute? _attribute;
            private Node[]? _children;
            private readonly TableInfo _tableInfo;
            private Attribute[]? _cachedAttributes;

            /// <summary>
            /// Creates a new tree node for the given data subset.
            /// </summary>
            /// <param name="tableInfo">Metadata about the training table.</param>
            /// <param name="data">The rows assigned to this node.</param>
            /// <param name="matchLabel">The partition label that led to this node, or null for the root.</param>
            public Node(TableInfo tableInfo, List<InMemoryRow> data, string? matchLabel)
            {
                _tableInfo = tableInfo;
                _classCount = data.GroupBy(d => d.ClassificationLabel).ToDictionary(g => g.Key, g => g.Count());
                Data = data;
                MatchLabel = matchLabel;
                Depth = 0;
            }

            /// <summary>
            /// Converts this internal node to a public DecisionTree.Node for the final model.
            /// </summary>
            /// <returns>A serializable decision tree node.</returns>
            public DecisionTree.Node AsDecisionTreeNode()
            {
                var ret = new DecisionTree.Node
                {
                    ColumnIndex = _attribute?.ColumnIndex,
                    MatchLabel = MatchLabel,
                    Split = _attribute?.Split,
                    Children = _children?.Select(c => c.AsDecisionTreeNode()).ToArray(),
                    Classification = PredictedClass
                };
                return ret;
            }

            /// <summary>
            /// Gets the candidate split attributes available at this node.
            /// Computed once and cached for subsequent accesses.
            /// </summary>
            public Attribute[] Attributes
            {
                get
                {
                    if (_cachedAttributes != null)
                        return _cachedAttributes;

                    var continuousValues = new Dictionary<uint, HashSet<double>>();
                    var categoricalValues = new Dictionary<uint, HashSet<string>>();
                    foreach (var item in Data)
                    {
                        foreach (var column in _tableInfo.CategoricalColumns)
                        {
                            if (!categoricalValues.TryGetValue(column, out var temp2))
                                categoricalValues.Add(column, temp2 = []);
                            temp2.Add(item.GetCategory(column));
                        }
                        foreach (var column in _tableInfo.ContinuousColumns)
                        {
                            if (!continuousValues.TryGetValue(column, out var temp))
                                continuousValues.Add(column, temp = []);
                            temp.Add(item.GetValue(column));
                        }
                    }

                    var ret = new HashSet<Attribute>();
                    foreach (var column in categoricalValues)
                    {
                        if (column.Value.Count > 1)
                        {
                            foreach (var item in column.Value)
                                ret.Add(new Attribute(column.Key, item));
                        }
                    }
                    foreach (var column in continuousValues)
                    {
                        if (column.Value.Count > 1)
                        {
                            var orderedContinuous = column.Value.OrderBy(v => v).ToList();
                            for (var i = 1; i < orderedContinuous.Count; i++)
                            {
                                var mid = (orderedContinuous[i - 1] + orderedContinuous[i]) / 2;
                                ret.Add(new Attribute(column.Key, mid));
                            }
                        }
                    }
                    _cachedAttributes = [.. ret];
                    return _cachedAttributes;
                }
            }

            /// <summary>
            /// Gets the entropy of the class distribution at this node.
            /// Entropy is calculated as H = -sum(p * log2(p)) for each class probability p.
            /// </summary>
            public double Entropy
            {
                get
                {
                    double total = _classCount.Sum(d => d.Value);
                    double ret = 0;
                    foreach (var item in _classCount)
                    {
                        var probability = item.Value / total;
                        if (probability > 0)
                            ret -= probability * Math.Log(probability, 2);
                    }
                    return ret;
                }
            }

            /// <summary>
            /// Sets the split attribute and children for this node, finalizing the node as an internal node.
            /// </summary>
            /// <param name="attribute">The attribute to split on.</param>
            /// <param name="children">The child nodes resulting from the split.</param>
            /// <returns>The children array.</returns>
            public Node[] SetAttribute(Attribute attribute, Node[] children)
            {
                _attribute = attribute;
                _children = children;
                foreach (var child in children)
                {
                    child._parent = this;
                    child.Depth = this.Depth + 1;
                }
                return children;
            }

            /// <summary>
            /// Gets the depth of this node in the tree (root is depth 0).
            /// Cached at assignment time for O(1) access.
            /// </summary>
            public int Depth { get; private set; }

            /// <summary>
            /// Gets the number of leaf nodes in the subtree rooted at this node.
            /// </summary>
            public int Leaves
            {
                get
                {
                    if (_children == null)
                        return 1;
                    else
                    {
                        var ret = 0;
                        foreach (var child in _children)
                            ret += child.Leaves;
                        return ret;
                    }
                }
            }

            /// <summary>
            /// Gets a value indicating whether this node is a leaf (pure or single-class).
            /// </summary>
            public bool IsLeaf => _classCount.Count <= 1;

            /// <summary>
            /// Gets the predicted class label for this node (the majority class).
            /// </summary>
            public string? PredictedClass => _classCount.OrderByDescending(kv => kv.Value).Select(kv => kv.Key).FirstOrDefault();

            /// <summary>
            /// Gets the training rows assigned to this node.
            /// </summary>
            public List<InMemoryRow> Data { get; }

            /// <summary>
            /// Gets the partition label that routes to this node, or null for the root.
            /// </summary>
            public string? MatchLabel { get; }
        }

        /// <summary>
        /// Configuration options for decision tree training.
        /// </summary>
        public class Config
        {
            /// <summary>
            /// Gets or sets the number of features to include in each random feature bag.
            /// When set, a random subset of this many features is considered at each split.
            /// Use with random forest training to decorrelate trees.
            /// </summary>
            public uint? FeatureBagCount { get; set; } = null;

            /// <summary>
            /// Gets or sets the minimum number of training samples required to attempt a split at a node.
            /// Nodes with fewer samples than this threshold become leaves.
            /// </summary>
            public int? MinDataPerNode { get; set; } = null;

            /// <summary>
            /// Gets or sets the maximum depth of the tree.
            /// Nodes at or beyond this depth become leaves.
            /// </summary>
            public int? MaxDepth { get; set; } = null;

            /// <summary>
            /// Gets or sets the minimum information gain required to perform a split.
            /// Splits that produce less gain than this threshold are discarded.
            /// </summary>
            public double? MinInformationGain { get; set; } = null;

            /// <summary>
            /// Gets or sets the maximum number of attributes to consider at each split.
            /// When set without FeatureBagCount, a random subset of attributes is selected.
            /// When set with FeatureBagCount, this value overrides the bag size.
            /// </summary>
            public uint? MaxAttributes { get; set; } = null;
        }

        /// <summary>
        /// Trains a decision tree on the provided data table.
        /// </summary>
        /// <param name="table">The training data table. Must have a target column set.</param>
        /// <param name="config">Optional configuration for controlling tree growth. Defaults to unrestricted growth.</param>
        /// <returns>A trained <see cref="DecisionTree"/> model.</returns>
        /// <exception cref="ArgumentNullException">Thrown when table is null.</exception>
        public static DecisionTree Train(IDataTable table, Config? config = null)
        {
            if (table == null)
                throw new ArgumentNullException(nameof(table));

            var tableInfo = new TableInfo(table);
            var root = new Node(tableInfo, tableInfo.Data, null);
            var stack = new Stack<Node>();
            stack.Push(root);

            var maxDepth = config?.MaxDepth;
            var minDataPerNode = config?.MinDataPerNode;
            var featureBagCount = config?.FeatureBagCount;
            double? minInformationGain = config?.MinInformationGain;
            var maxAttributes = config?.MaxAttributes;

            while (stack.Any())
            {
                var node = stack.Pop();

                // stop at leaf nodes
                if (node.IsLeaf)
                    continue;

                // stop when there are no more features left to split
                var attributes = node.Attributes;
                if (!attributes.Any())
                    continue;

                // stop when max depth is exceeded
                if (maxDepth.HasValue && node.Depth >= maxDepth.Value)
                    continue;

                // stop when the node has too few samples
                if (minDataPerNode.HasValue && node.Data.Count < minDataPerNode.Value)
                    continue;

                // limit the number of attributes considered at this split
                if (featureBagCount.HasValue)
                {
                    // Use FeatureBagCount as the default bag size, overridden by MaxAttributes if both are set
                    var bagSize = maxAttributes ?? featureBagCount.Value;
                    attributes = attributes.Bag(bagSize, table.Context.Random);
                }
                else if (maxAttributes.HasValue)
                {
                    // Randomly select a subset of attributes without replacement
                    attributes = [.. attributes.Shuffle(table.Context.Random).Take((int)maxAttributes.Value)];
                }

                var nodeEntropy = node.Entropy;
                double nodeTotal = node.Data.Count;
                var scoreTable = new List<(Attribute Attribute, Node[] Nodes, double Score)>();
                foreach (var item in attributes)
                {
                    var newChildren = item.Partition(node.Data).Select(d => new Node(tableInfo, d.Value, d.Key)).ToArray();
                    var informationGain = GetInformationGain(nodeEntropy, nodeTotal, newChildren);
                    if (informationGain < minInformationGain)
                        continue;
                    scoreTable.Add((item, newChildren, informationGain));
                }

                if (scoreTable.Any())
                {
                    var bestSplit = scoreTable.MaxBy(kv => kv.Score);
                    foreach (var child in node.SetAttribute(bestSplit.Attribute, bestSplit.Nodes))
                        stack.Push(child);
                }
            }

            return new DecisionTree
            {
                ClassColumnIndex = tableInfo.ClassColumnIndex,
                Root = root.AsDecisionTreeNode()
            };
        }

        /// <summary>
        /// Calculates the weighted information gain achieved by splitting a node into children.
        /// Information gain is the parent entropy minus the weighted average of child entropies.
        /// </summary>
        /// <param name="setEntropy">The entropy of the parent node.</param>
        /// <param name="setCount">The total number of samples in the parent node.</param>
        /// <param name="splits">The child nodes produced by the split.</param>
        /// <returns>The information gain from the split.</returns>
        static double GetInformationGain(double setEntropy, double setCount, Node[] splits)
        {
            var weightedEntropy = setEntropy;
            foreach (var item in splits)
            {
                weightedEntropy -= item.Data.Count / setCount * item.Entropy;
            }
            return weightedEntropy;
        }
    }
}
