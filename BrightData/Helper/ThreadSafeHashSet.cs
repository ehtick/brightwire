using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Threading;

namespace BrightData.Helper
{
    /// <summary>
    /// A hash set that can be used by more than one thread
    /// </summary>
    /// <typeparam name="T">The wrapped type</typeparam>
    public sealed class ThreadSafeHashSet<T>(int? capacity = null) : IEnumerable<T>, IDisposable
    {
        readonly ReaderWriterLockSlim _lock = new(LockRecursionPolicy.NoRecursion);
        readonly HashSet<T> _hashSet = capacity.HasValue ? new HashSet<T>(capacity.Value) : new HashSet<T>();
        bool _disposed = false;

        /// <inheritdoc />
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _hashSet.Clear();
            _lock.Dispose();
            GC.SuppressFinalize(this);
        }

        void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ThreadSafeHashSet<T>));
        }

        /// <summary>
        /// Adds a new item
        /// </summary>
        /// <param name="item">Item to add</param>
        /// <returns></returns>
        public bool Add(T item)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                return _hashSet.Add(item);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Adds all items from the collection in a single lock acquisition
        /// </summary>
        /// <param name="items">Items to add</param>
        /// <returns>Number of items that were newly added</returns>
        public int AddRange(IEnumerable<T> items)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                int added = 0;
                foreach (var item in items)
                {
                    if (_hashSet.Add(item))
                        added++;
                }
                return added;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Clears all items
        /// </summary>
        public void Clear()
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                _hashSet.Clear();
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Checks if the set contains the specified item
        /// </summary>
        /// <param name="item">Item to find</param>
        /// <returns></returns>
        public bool Contains(T item)
        {
            ThrowIfDisposed();
            _lock.EnterReadLock();
            try
            {
                return _hashSet.Contains(item);
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>
        /// Checks if the set contains all items in the collection
        /// </summary>
        /// <param name="items">Items to check</param>
        /// <returns>True if all items are present</returns>
        public bool ContainsAll(IEnumerable<T> items)
        {
            ThrowIfDisposed();
            _lock.EnterReadLock();
            try
            {
                foreach (var item in items)
                {
                    if (!_hashSet.Contains(item))
                        return false;
                }
                return true;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>
        /// Removes the items that match the predicate in a single lock acquisition
        /// </summary>
        /// <param name="predicate">Predicate to match</param>
        /// <returns>Number of items removed</returns>
        public int RemoveWhere(Predicate<T> predicate)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                var toRemove = new List<T>();
                foreach (var item in _hashSet)
                {
                    if (predicate(item))
                        toRemove.Add(item);
                }

                int removed = 0;
                foreach (var item in toRemove)
                {
                    if (_hashSet.Remove(item))
                        removed++;
                }
                return removed;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Removes an item
        /// </summary>
        /// <param name="item">Item to remove</param>
        /// <returns>True if the item was removed</returns>
        public bool Remove(T item)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                return _hashSet.Remove(item);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// The number of items in the set
        /// </summary>
        public int Count
        {
            get
            {
                ThrowIfDisposed();
                _lock.EnterReadLock();
                try
                {
                    return _hashSet.Count;
                }
                finally
                {
                    _lock.ExitReadLock();
                }
            }
        }

        /// <summary>
        /// Applies a callback to each item in the set.
        /// Note: A snapshot is taken under the lock, then iterated outside the lock
        /// to avoid blocking writers during long-running callbacks.
        /// </summary>
        /// <param name="callback">Callback to invoke for each item</param>
        public void ForEach(Action<T> callback)
        {
            ThrowIfDisposed();
            T[] snapshot;

            _lock.EnterReadLock();
            try
            {
                snapshot = new T[_hashSet.Count];
                _hashSet.CopyTo(snapshot, 0);
            }
            finally
            {
                _lock.ExitReadLock();
            }

            foreach (var item in snapshot)
                callback(item);
        }

        /// <summary>
        /// Tries to pop an item from the set
        /// </summary>
        /// <param name="ret">Item that was removed</param>
        /// <returns>True if there was an item to remove</returns>
        public bool TryPop([MaybeNullWhen(false)] out T ret)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                var enumerator = _hashSet.GetEnumerator();
                if (enumerator.MoveNext())
                {
                    ret = enumerator.Current;
                    return _hashSet.Remove(ret);
                }

                ret = default;
                return false;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Modifies the set to contain all elements that are contained in the set, in the other collection, or both.
        /// </summary>
        /// <param name="other">Collection to union with</param>
        public void UnionWith(IEnumerable<T> other)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                foreach (var item in other)
                    _hashSet.Add(item);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Modifies the set to contain only elements that are also in the specified collection.
        /// </summary>
        /// <param name="other">Collection to intersect with</param>
        public void IntersectWith(IEnumerable<T> other)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                var otherSet = new HashSet<T>(other);
                _hashSet.IntersectWith(otherSet);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Removes all elements of the specified collection from the set.
        /// </summary>
        /// <param name="other">Collection to exclude</param>
        public void ExceptWith(IEnumerable<T> other)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                var otherSet = new HashSet<T>(other);
                _hashSet.ExceptWith(otherSet);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Modifies the set to contain only elements that are present in either the set or the specified collection, but not both.
        /// </summary>
        /// <param name="other">Collection to symmetrically exclude with</param>
        public void SymmetricExceptWith(IEnumerable<T> other)
        {
            ThrowIfDisposed();
            _lock.EnterWriteLock();
            try
            {
                var otherSet = new HashSet<T>(other);
                _hashSet.SymmetricExceptWith(otherSet);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Determines whether the set is a subset of the specified collection.
        /// </summary>
        /// <param name="other">Collection to compare against</param>
        /// <returns>True if this set is a subset of other</returns>
        public bool IsSubsetOf(IEnumerable<T> other)
        {
            ThrowIfDisposed();
            _lock.EnterReadLock();
            try
            {
                var otherSet = new HashSet<T>(other);
                return _hashSet.IsSubsetOf(otherSet);
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>
        /// Determines whether the set is a superset of the specified collection.
        /// </summary>
        /// <param name="other">Collection to compare against</param>
        /// <returns>True if this set is a superset of other</returns>
        public bool IsSupersetOf(IEnumerable<T> other)
        {
            ThrowIfDisposed();
            _lock.EnterReadLock();
            try
            {
                foreach (var item in other)
                {
                    if (!_hashSet.Contains(item))
                        return false;
                }
                return true;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>
        /// Returns a snapshot copy of the items in the set
        /// </summary>
        /// <returns>Array containing all items</returns>
        public T[] ToArray()
        {
            ThrowIfDisposed();
            _lock.EnterReadLock();
            try
            {
                var result = new T[_hashSet.Count];
                _hashSet.CopyTo(result, 0);
                return result;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <inheritdoc />
        public IEnumerator<T> GetEnumerator()
        {
            return ((IEnumerable<T>)ToArray()).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
